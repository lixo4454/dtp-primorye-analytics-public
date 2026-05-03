"""Страница «Предсказатель тяжести ДТП» — POST /predict/severity.

UX-стратегия — 34 признака разнесены по 6 тематическим вкладкам, чтобы
форма не выглядела как стена полей:

    Время · Место · Условия · ТС и участники · Поведение · Тип ДТП

Дефолтные значения предсчитаны офлайн (см.
``scripts/build_catboost_form_schema.py``):
- median для числовых → центральный валидный профиль ДТП
- mode для категориальных → самый частый вариант
- top-30 значений для каждой категориальной колонки → допустимые опции

Submit вызывает ``predict_severity`` через POST /predict/severity →
4 калиброванные вероятности (изотонная per-class с per-class isotonic). Калибровка
гарантирует ECE < 0.05 для всех 4 классов — числам можно доверять
без post-hoc bias-correction.

Counterfactual («что если бы…») оставлен на отдельную секцию (Recommendations
Engine) — в этой странице делаем точную одну точку прогноза.
"""

from __future__ import annotations

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.utils.api_client import predict_severity, predict_severity_counterfactual
from src.app.utils.defaults import (
    CLASS_COLORS,
    CLASS_LABELS_RU,
    FEATURE_LABELS,
    FORM_GROUPS,
    load_form_schema,
)
from src.app.utils.styling import page_footer, page_header
from src.app.utils.visualizations import fmt_int

page_header(
    title="Предсказатель тяжести ДТП",
    subtitle=(
        "CatBoost v2 (Optuna, 200 trials) + изотонная калибровка per-class — "
        "выдаёт **калиброванные** вероятности 4 классов исходов. "
        "ECE < 0.05 на test-выборке для всех классов."
    ),
    icon=":material/sensors:",
)

# CTA для впервые открывших страницу: 34 поля формы могут показаться
# ошеломляющими. Поясняем что предзаполненные дефолты — валидный baseline,
# и можно сразу нажать «Предсказать».
st.info(
    """
    **Не знаешь что заполнять?** Все 34 поля формы предзаполнены **медианой
    и модой обучающей выборки** (23 530 ДТП Приморского края): это валидный
    baseline-профиль типичного ДТП в регионе. Можно нажать «Предсказать тяжесть»
    сразу — получишь baseline-предсказание (P(dead) ≈ class prior).
    Затем меняй конкретные поля (`unbelted_share`, `traffic_area_state`,
    `is_highway` и т. д.) и смотри как меняются вероятности.
    """,
    icon=":material/lightbulb:",
)

schema = load_form_schema()
defaults = schema["defaults"]
cat_choices = schema["cat_choices"]
ranges = schema["numeric_ranges"]
cat_features = schema["cat_features"]
bool_features = schema["bool_features"]


# ============================================================
# session_state init / reset
# ============================================================
def _init_form_state() -> None:
    """Кладёт дефолты в session_state.<key> на cold start."""
    for k, v in defaults.items():
        sk = f"sev_{k}"
        if sk not in st.session_state:
            st.session_state[sk] = v


def _reset_form() -> None:
    """Перетирает все sev_* ключи дефолтами. Перезаписать значение
    в session_state — единственный способ заставить Streamlit
    подхватить программное изменение поля формы."""
    for k, v in defaults.items():
        st.session_state[f"sev_{k}"] = v


_init_form_state()


# ============================================================
# Class priors на train — отдельным info-блоком одной строкой,
# чтобы не отвлекать внимание от формы вверху страницы.
# ============================================================
priors = schema["class_priors"]
st.caption(
    f":material/info: **Class priors на train (23 530 ДТП):** "
    f"Лёгкая **{priors.get('light', 0) * 100:.1f} %** · "
    f"Тяжёлая **{priors.get('severe', 0) * 100:.1f} %** · "
    f"Тяжёлая (множ.) **{priors.get('severe_multiple', 0) * 100:.1f} %** · "
    f"Смертельная **{priors.get('dead', 0) * 100:.1f} %**. "
    f"Это базовый уровень — модель сравнивает с ним свои предсказания "
    f"для оценки relative risk."
)


# ============================================================
# Form widgets per group
# ============================================================
def _render_field(col_name: str) -> None:
    """Рендерит один input нужного типа в зависимости от колонки."""
    sk = f"sev_{col_name}"
    label = FEATURE_LABELS.get(col_name, col_name)

    if col_name in cat_features:
        opts = cat_choices.get(col_name, [defaults[col_name]])
        if defaults[col_name] not in opts:
            opts = [defaults[col_name], *opts]
        st.selectbox(label, opts, key=sk)
    elif col_name in bool_features:
        st.checkbox(label, key=sk)
    else:
        rng = ranges.get(col_name, {})
        # Целочисленные: hour/dow/month/year/veh_amount/...
        if col_name in {
            "hour",
            "dow",
            "month",
            "year",
            "veh_amount",
            "veh_count_actual",
            "part_count",
            "ped_count",
        }:
            st.number_input(
                label,
                min_value=int(rng.get("min", 0)),
                max_value=int(rng.get("max", 100)),
                step=1,
                key=sk,
            )
        else:
            # Float
            st.number_input(
                label,
                min_value=float(rng.get("min", 0.0)),
                max_value=float(rng.get("max", 1.0)),
                step=(0.01 if "share" in col_name else 1.0 if "year" in col_name else 0.1),
                format=(
                    "%.3f"
                    if col_name in {"lat", "lon"}
                    else "%.2f"
                    if "share" in col_name
                    else "%.1f"
                ),
                key=sk,
            )


# ============================================================
# Tabs — 6 семантических групп
# ============================================================
tab_names = list(FORM_GROUPS.keys())
tabs = st.tabs(tab_names)

for tab, group_name in zip(tabs, tab_names):
    with tab:
        cols = st.columns(2)
        fields = FORM_GROUPS[group_name]
        for idx, col_name in enumerate(fields):
            with cols[idx % 2]:
                _render_field(col_name)


st.divider()

# ============================================================
# Submit + Reset — рядом, на одной линии
# ============================================================
predict_col, reset_col, _spacer = st.columns([1.4, 1.2, 2.4])
do_predict = predict_col.button(
    ":material/sensors: Предсказать тяжесть",
    type="primary",
    use_container_width=True,
)
reset_col.button(
    ":material/restart_alt: Сбросить к дефолтам",
    on_click=_reset_form,
    use_container_width=True,
    help=(
        "Возвращает все 34 поля к медиане/моде обучающей выборки. "
        "Удобно после серии экспериментов — быстро вернуться к baseline."
    ),
)


def _build_payload() -> dict:
    """Собирает 34-полевой payload из session_state."""
    return {k: st.session_state[f"sev_{k}"] for k in defaults}


if do_predict:
    payload = _build_payload()
    try:
        with st.spinner("CatBoost + isotonic calibration..."):
            response = predict_severity(payload)
    except httpx.HTTPError as exc:
        st.error(f"FastAPI вернул ошибку: {exc}")
        if hasattr(exc, "response") and exc.response is not None:
            try:
                st.code(exc.response.text)
            except Exception:
                pass
        st.stop()

    probs = response["probabilities"]
    predicted = response["predicted_class"]

    # ============================================================
    # Результат: 4 metric + bar chart + интерпретация
    # ============================================================
    st.subheader("Результат")

    rcols = st.columns(4)
    for i, cls in enumerate(["light", "severe", "severe_multiple", "dead"]):
        with rcols[i]:
            p = probs.get(cls, 0)
            label = CLASS_LABELS_RU[cls]
            star = " :material/star:" if cls == predicted else ""
            st.metric(label + star, f"{p * 100:.2f}%")

    # Bar chart с цветовой схемой severity
    bar_df = pd.DataFrame(
        [
            {
                "class": CLASS_LABELS_RU[c],
                "prob": probs.get(c, 0),
                "color": CLASS_COLORS[c],
            }
            for c in ["light", "severe", "severe_multiple", "dead"]
        ]
    )
    fig = go.Figure(
        go.Bar(
            x=bar_df["class"],
            y=bar_df["prob"],
            marker_color=bar_df["color"],
            text=[f"{p * 100:.1f}%" for p in bar_df["prob"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>P = %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Калиброванная вероятность",
        yaxis=dict(tickformat=".0%"),
        plot_bgcolor="#f5f7fa",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # Интерпретация
    # ============================================================
    p_dead = probs.get("dead", 0)
    p_light = probs.get("light", 0)
    base_dead = priors.get("dead", 0.087)
    rel_to_base = p_dead / base_dead if base_dead else 1

    if p_dead >= 0.4:
        st.error(
            f"**Высокий риск смертельного исхода — {p_dead * 100:.1f}%** "
            f"(в {rel_to_base:.1f}× выше базового уровня {base_dead * 100:.1f}%). "
            f"Бинарный приоритетный флаг диспетчера 112 (порог 0.4)."
        )
    elif p_dead >= 0.2:
        st.warning(
            f"**Повышенный риск смертельного исхода — {p_dead * 100:.1f}%** "
            f"(в {rel_to_base:.1f}× выше базового уровня)."
        )
    elif p_light >= 0.7:
        st.success(
            f"**Доминирует «лёгкая» — {p_light * 100:.1f}%.** "
            f"Вероятность смертельного {p_dead * 100:.2f}% — близко к "
            f"базовому уровню."
        )
    else:
        st.info(
            f"Mixed-prediction: смертельная {p_dead * 100:.2f}%, "
            f"тяжёлая {probs.get('severe', 0) * 100:.2f}%. "
            f"Predicted class = «{CLASS_LABELS_RU[predicted]}»."
        )

    with st.expander(":material/data_object: Payload и raw-ответ FastAPI"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Request → POST /predict/severity**")
            st.json(payload)
        with c2:
            st.markdown("**Response**")
            st.json(response)

    st.caption(
        f"_Калибровка: {response.get('calibration_method')}, "
        f"модель: {response.get('model_version')}_"
    )
else:
    st.info(
        f"Нажми **«Предсказать тяжесть»** — текущие значения формы "
        f"(дефолты или твои правки) уйдут POST'ом на FastAPI. "
        f"34 признака · {fmt_int(schema['n_train'])} ДТП в обучающей выборке."
    )


# ====================================================================
# Симуляция мер (counterfactual через CatBoost) — отдельная крупная секция
# ====================================================================
# Визуальный разрыв (двойной divider) — это новая страница по содержанию,
# не продолжение формы выше.
st.divider()
st.markdown(
    "<h2 style='margin-top: 1.5rem;'>" ":material/science: Симуляция мер (counterfactual)" "</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='dtp-section-lead'>Per-incident оценка изменения вероятностей "
    "при применении мер к этому конкретному ДТП. Это <b>marginal effect</b> "
    "калиброванной CatBoost-модели, не population-average CMF из литературы. "
    "См. <code>docs/recommendations_methodology.md §4.3</code> — counterfactual "
    "и rule engine дополняют друг друга.</p>",
    unsafe_allow_html=True,
)

# 7 чекбоксов мер (см. methodology §4.2)
COUNTERFACTUAL_MEASURES = [
    {
        "key": "belt",
        "label": "Все участники пристёгнуты",
        "overrides": {"unbelted_share": 0.0},
        "icon": ":material/airline_seat_recline_normal:",
    },
    {
        "key": "drunk",
        "label": "Никто не пьян",
        "overrides": {"drunk_share": 0.0},
        "icon": ":material/no_drinks:",
    },
    {
        "key": "light",
        "label": "Освещение перехода (ночь → ночь+освещ.)",
        "overrides": {"light_type": "В темное время суток, освещение включено"},
        "icon": ":material/lightbulb:",
    },
    {
        "key": "state",
        "label": "Сухое покрытие (вместо мокрого/гололёда)",
        "overrides": {"traffic_area_state": "Сухое"},
        "icon": ":material/blur_on:",
    },
    {
        "key": "defects",
        "label": "Дефекты дороги устранены",
        "overrides": {"has_defect": False},
        "icon": ":material/build:",
    },
    {
        "key": "highway",
        "label": "Перевод участка в гор. улицу (трасса → город)",
        "overrides": {"is_highway": False},
        "icon": ":material/location_city:",
    },
    {
        "key": "all",
        "label": "Все вышеперечисленные одновременно",
        "overrides": {
            "unbelted_share": 0.0,
            "drunk_share": 0.0,
            "light_type": "В темное время суток, освещение включено",
            "traffic_area_state": "Сухое",
            "has_defect": False,
            "is_highway": False,
        },
        "icon": ":material/checklist:",
    },
]

cf_cols = st.columns(2)
for i, m in enumerate(COUNTERFACTUAL_MEASURES):
    with cf_cols[i % 2]:
        st.checkbox(
            f"{m['icon']} {m['label']}",
            key=f"cf_{m['key']}",
            value=(m["key"] in {"belt", "drunk"}),  # дефолт: belt+drunk ON
        )

# Подсчитываем выбранные ДО рендера кнопки — используем для disabled-state
_selected = [m for m in COUNTERFACTUAL_MEASURES if st.session_state.get(f"cf_{m['key']}", False)]

btn_col, hint_col = st.columns([1, 3])
with btn_col:
    run_cf = st.button(
        ":material/play_arrow: Запустить симуляцию",
        type="primary",
        use_container_width=True,
        disabled=(len(_selected) == 0),
        help=(
            "Выбери хотя бы одну меру выше"
            if len(_selected) == 0
            else f"Будет симулировано {len(_selected)} сценариев"
        ),
    )
with hint_col:
    if len(_selected) == 0:
        st.caption(":material/info: Отметь хотя бы один чекбокс — кнопка станет активной.")
    else:
        names = ", ".join(f"«{m['label']}»" for m in _selected[:3])
        if len(_selected) > 3:
            names += f" и ещё {len(_selected) - 3}"
        st.caption(f":material/check: К запуску: {names}")

if run_cf:
    selected = _selected  # совпадает с состоянием на момент клика

    payload = _build_payload()
    cf_request = {
        "baseline": payload,
        "scenarios": [{"name": m["label"], "overrides": m["overrides"]} for m in selected],
    }
    try:
        with st.spinner(f"Симулирую {len(selected)} сценариев..."):
            cf_response = predict_severity_counterfactual(cf_request)
    except httpx.HTTPError as exc:
        st.error(f"FastAPI вернул ошибку: {exc}")
        if hasattr(exc, "response") and exc.response is not None:
            try:
                st.code(exc.response.text)
            except Exception:
                pass
        st.stop()

    base_proba = cf_response["baseline_proba"]
    base_dead = base_proba.get("dead", 0.0)

    # Топ-метрики: baseline + лучший сценарий
    best_scenario = min(
        cf_response["scenarios"], key=lambda s: s["modified_proba"].get("dead", 1.0)
    )

    mc_cols = st.columns(3)
    mc_cols[0].metric(
        "Baseline P(dead)",
        f"{base_dead * 100:.2f}%",
        help="Вероятность смертельного исхода для исходного профиля",
    )
    mc_cols[1].metric(
        f"Лучший сценарий: {best_scenario['name']}",
        f"{best_scenario['modified_proba'].get('dead', 0) * 100:.2f}%",
        delta=f"{best_scenario['delta_dead_pct_points']:.2f} п.п.",
        delta_color="inverse",  # отрицательная Δ = зелёный
    )
    mc_cols[2].metric(
        "Снижение P(dead)",
        f"{abs(best_scenario['delta_dead_pct_points']):.2f} п.п.",
        f"−{abs(best_scenario['delta_dead_pct_points'] / (base_dead * 100) * 100):.0f}% относ.",
        delta_color="inverse",
    )

    # Waterfall: baseline → −Δ1 → −Δ2 →... → final
    # Каждый сценарий применяется отдельно — это не chain, а параллель,
    # но waterfall удобен визуально для сравнения relative effects.
    st.markdown("**Сравнение сценариев — Waterfall (относительно baseline P(dead))**")

    # Сортируем по убыванию delta_dead (наиболее эффективные слева)
    sorted_scenarios = sorted(cf_response["scenarios"], key=lambda s: s["delta_dead_pct_points"])

    # Plotly Waterfall: показываем baseline как первый bar,
    # затем каждый сценарий как «измерение от baseline»
    measures = ["absolute"] + ["relative"] * len(sorted_scenarios)
    x_labels = ["Baseline"] + [s["name"][:30] for s in sorted_scenarios]
    # Значения: для absolute — base_dead*100, для relative — delta
    y_values = [base_dead * 100] + [s["delta_dead_pct_points"] for s in sorted_scenarios]

    fig_wf = go.Figure(
        go.Waterfall(
            measure=measures,
            x=x_labels,
            y=y_values,
            text=[f"{base_dead * 100:.2f}%" + "".join("" for _ in range(0))]
            + [f"{s['delta_dead_pct_points']:+.2f} п.п." for s in sorted_scenarios],
            textposition="outside",
            connector={"line": {"color": "rgb(150,150,150)"}},
            decreasing={"marker": {"color": "#2ca02c"}},  # зелёный = снижение риска
            increasing={"marker": {"color": "#d62728"}},  # красный = рост (не должно быть)
            totals={"marker": {"color": "#1f77b4"}},
        )
    )
    fig_wf.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=120),
        yaxis_title="P(dead), %",
        plot_bgcolor="#f5f7fa",
        xaxis_tickangle=-25,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

    # Таблица всех 4 классов × все сценарии
    st.markdown("**Все классы — детальная таблица**")
    rows = [
        {
            "Сценарий": "Baseline",
            **{
                CLASS_LABELS_RU[c]: f"{base_proba.get(c, 0) * 100:.2f}%"
                for c in ["light", "severe", "severe_multiple", "dead"]
            },
        }
    ]
    for s in cf_response["scenarios"]:
        rows.append(
            {
                "Сценарий": s["name"],
                **{
                    CLASS_LABELS_RU[c]: f"{s['modified_proba'].get(c, 0) * 100:.2f}%"
                    for c in ["light", "severe", "severe_multiple", "dead"]
                },
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Интерпретация: какие меры работают для этого профиля
    significant = [s for s in cf_response["scenarios"] if s["delta_dead_pct_points"] <= -1.0]
    if significant:
        meta = ", ".join(
            f"**{s['name']}** ({s['delta_dead_pct_points']:+.1f} п.п.)" for s in significant[:3]
        )
        st.success(
            f"Для этого профиля наиболее эффективны: {meta}. Модель показывает "
            f"снижение P(dead) на ≥1 п.п. — это marginal effect, не causal."
        )
    else:
        st.info(
            "Эффекты выбранных мер для этого профиля невелики (<1 п.п.). "
            "Проверь baseline — возможно, базовый риск уже низкий, или "
            "профиль out-of-distribution."
        )

    with st.expander(":material/data_object: Counterfactual raw-ответ"):
        st.json(cf_response)

# Footer — единый для всех 7 страниц.
page_footer()
