"""
Analysis Tab — Deep strategy analytics with Monte Carlo, Prop Firm sim, etc.
Called from app.py. All computation is in analytics.py.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics import (
    monthly_returns,
    risk_performance_stats,
    distribution_stats,
    efficiency_stats,
    time_analysis,
    monte_carlo,
    prop_firm_simulation,
)


def render_analysis_tab(
    list_strategies_fn,
    get_strategy_fn,
    load_params_fn,
    save_params_fn,
    load_from_yfinance_fn,
    load_from_csv_fn,
    run_bt_generic_fn,
    metric_card_fn,
    fmt_dollar_fn,
    color_for_fn,
    data_source,
    uploaded_file,
    timeframe,
    ticker,
):
    """Render the full Analysis tab inside the Streamlit app."""

    st.markdown("### Strategy Analysis")
    st.caption(
        "Run any strategy and get deep analytics: distribution, efficiency, "
        "time-of-day, Monte Carlo simulation, and prop firm challenge testing."
    )

    # ── Strategy & Data Selection ─────────────────────────────────────────────
    an_strat_list = list_strategies_fn()
    an_strat_names = {k: name for k, name, _ in an_strat_list}
    an_strat_descs = {k: desc for k, _, desc in an_strat_list}

    an_col1, an_col2 = st.columns([2, 1])
    with an_col1:
        an_strategy_key = st.selectbox(
            "Select Strategy",
            options=[k for k, _, _ in an_strat_list],
            format_func=lambda k: an_strat_names[k],
            key="an_strat_select",
        )
        st.caption(an_strat_descs[an_strategy_key])
    with an_col2:
        an_date_opt = st.selectbox(
            "Date Range",
            ["Last 30 days", "Last 90 days", "Last 365 days", "Last 2 years", "Custom"],
            index=2,
            key="an_date_range",
        )

    from datetime import date

    if an_date_opt == "Custom":
        _ac1, _ac2 = st.columns(2)
        with _ac1:
            an_start = st.date_input("Start", value=date.today() - pd.Timedelta(days=365), key="an_start")
        with _ac2:
            an_end = st.date_input("End", value=date.today(), key="an_end")
    else:
        _dmap = {"Last 30 days": 30, "Last 90 days": 90, "Last 365 days": 365, "Last 2 years": 730}
        an_start = date.today() - pd.Timedelta(days=_dmap[an_date_opt])
        an_end = date.today()

    an_active_strat = get_strategy_fn(an_strategy_key)
    _an_saved = load_params_fn(an_strategy_key)

    an_capital = st.number_input("Capital ($)", value=int(_an_saved.get("initial_capital", 50000)), step=1000, key="an_cap")

    if st.button("Run Analysis", type="primary", key="an_run"):
        # Load data — handle all source types
        df_an = pd.DataFrame()
        if data_source == "Upload CSV/XLSX":
            if uploaded_file is None:
                st.error("Upload a data file in the sidebar first.")
                st.stop()
            df_an = load_from_csv_fn(uploaded_file)
        elif "MNQ 1-Min" in data_source:
            # Load built-in MNQ data
            import os
            _mnq_path = os.path.join(os.path.dirname(__file__), "data", "MNQ_1min_continuous.csv")
            if os.path.exists(_mnq_path):
                df_an = pd.read_csv(_mnq_path, index_col=0, parse_dates=True)
                if df_an.index.tz is None:
                    df_an.index = df_an.index.tz_localize("UTC")
                df_an.index = df_an.index.tz_convert("US/Eastern")
                df_an = df_an.dropna(subset=["open", "high", "low", "close"])
            else:
                st.error("Built-in MNQ data file not found.")
                st.stop()
        else:
            with st.spinner(f"Fetching {timeframe} data for {ticker}..."):
                df_an = load_from_yfinance_fn(ticker, interval=timeframe)

        if df_an.empty:
            st.error("No data available. Check your data source in the sidebar.")
            st.stop()

        # Date filter
        _s = pd.Timestamp(an_start)
        _e = pd.Timestamp(an_end) + pd.Timedelta(days=1)
        if df_an.index.tz is not None:
            _s = _s.tz_localize(df_an.index.tz)
            _e = _e.tz_localize(df_an.index.tz)
        df_an = df_an.loc[_s:_e]

        if df_an.empty:
            st.error("No data in selected range.")
            st.stop()

        # Build params
        an_params = {**an_active_strat.frozen_params(), **_an_saved, "initial_capital": float(an_capital), "point_value": 2.0}

        with st.spinner("Running backtest..."):
            try:
                an_result = run_bt_generic_fn(df_an, an_strategy_key, an_params)
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.stop()

        st.session_state["an_result"] = {
            "result": an_result,
            "capital": float(an_capital),
            "strategy_key": an_strategy_key,
            "strategy_name": an_strat_names[an_strategy_key],
        }

    # ── Display Results ───────────────────────────────────────────────────────
    if "an_result" not in st.session_state:
        st.info("Select a strategy and click **Run Analysis** to get started.")
        return

    _anr = st.session_state["an_result"]
    an_result = _anr["result"]
    an_capital = _anr["capital"]
    trades = an_result.get("trades", [])
    equity = an_result.get("equity", pd.Series(dtype=float))

    if not trades:
        st.warning("No trades produced. Try a different strategy or date range.")
        return

    st.success(f"**{_anr['strategy_name']}** — {len(trades)} trade records loaded")

    # Sub-tabs
    t_overview, t_dist, t_eff, t_time, t_mc, t_prop, t_trades = st.tabs([
        "Overview", "Distribution", "Efficiency", "Time", "Monte Carlo", "Prop Firm", "Trades"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # OVERVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with t_overview:
        rps = risk_performance_stats(trades, equity)

        # Win/Loss Analysis + Risk & Performance side by side
        ov_col1, ov_col2 = st.columns(2)
        with ov_col1:
            st.markdown("#### Win/Loss Analysis")
            _ov_data = {
                "Total Trades": f"{rps['total_trades']}",
                "Winning Trades": f"{rps['winning_trades']}",
                "Losing Trades": f"{rps['losing_trades']}",
                "Average Win": f"${rps['avg_win']:,.2f}",
                "Average Loss": f"${rps['avg_loss']:,.2f}",
                "Largest Win": f"${rps['largest_win']:,.2f}",
                "Largest Loss": f"${rps['largest_loss']:,.2f}",
            }
            for k, v in _ov_data.items():
                _c = "green" if "Win" in k else ("red" if "Loss" in k or "Losing" in k else "")
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;'>"
                            f"<span>{k}</span><span style='color:{_c};font-weight:bold'>{v}</span></div>",
                            unsafe_allow_html=True)

        with ov_col2:
            st.markdown("#### Risk & Performance")
            _rp_data = {
                "Max Consecutive Wins": (f"{rps['max_consecutive_wins']}", "green"),
                "Max Consecutive Losses": (f"{rps['max_consecutive_losses']}", "red"),
                "Sortino Ratio": (f"{rps['sortino_ratio']:.2f}", "green" if rps['sortino_ratio'] > 0 else "red"),
                "Recovery Factor": (f"{rps['recovery_factor']:.2f}", "green"),
                "SQN": (f"{rps['sqn']:.2f}", "green" if rps['sqn'] > 0 else "red"),
                "Expectancy": (f"${rps['expectancy']:,.2f}", "green" if rps['expectancy'] > 0 else "red"),
                "Avg Trade P&L": (f"${rps['avg_trade_pnl']:,.2f}", "green" if rps['avg_trade_pnl'] > 0 else "red"),
            }
            for k, (v, c) in _rp_data.items():
                st.markdown(f"<div style='display:flex;justify-content:space-between;padding:4px 0;'>"
                            f"<span>{k}</span><span style='color:{c};font-weight:bold'>{v}</span></div>",
                            unsafe_allow_html=True)

        # Monthly Returns Heatmap
        st.markdown("---")
        st.markdown("#### Monthly Returns")
        mr = monthly_returns(trades, an_capital)
        if not mr.empty:
            pivot = mr.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum").fillna(0)
            # Reorder months
            month_order = list(range(1, 13))
            pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
            month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                           7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
            pivot.columns = [month_names.get(c, c) for c in pivot.columns]

            # Add total column
            pivot["Total"] = pivot.sum(axis=1)

            # Build heatmap
            z = pivot.values
            text = [[f"+{v:,.0f}" if v > 0 else f"{v:,.0f}" if v != 0 else "–" for v in row] for row in z]

            # Color scale: red for negative, gray for zero, green for positive
            fig_mr = go.Figure(go.Heatmap(
                z=z, x=list(pivot.columns), y=[str(y) for y in pivot.index],
                text=text, texttemplate="%{text}", textfont=dict(size=12),
                colorscale=[[0, "#ff5252"], [0.5, "#1a1e2e"], [1, "#00e676"]],
                zmid=0, showscale=False,
            ))
            fig_mr.update_layout(
                template="plotly_dark", height=max(150, 60 * len(pivot)),
                margin=dict(l=60, r=30, t=10, b=30),
                xaxis=dict(side="top"),
            )
            st.plotly_chart(fig_mr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION
    # ══════════════════════════════════════════════════════════════════════════
    with t_dist:
        ds = distribution_stats(trades)
        pnl_vals = np.array(ds["pnl_values"])

        _mean_clr = "green" if ds["mean"] > 0 else "red"
        _med_clr = "green" if ds["median"] > 0 else "red"
        _is_normal = abs(ds["skewness"]) < 1 and abs(ds["kurtosis"]) < 3
        _norm_clr = "green" if _is_normal else "red"
        _norm_txt = "Yes" if _is_normal else "No (fat tails)"

        st.markdown(
            f"**Mean:** <span style='color:{_mean_clr}'>${ds['mean']:,.2f}</span> &nbsp;&nbsp; "
            f"**Median:** <span style='color:{_med_clr}'>${ds['median']:,.2f}</span> &nbsp;&nbsp; "
            f"**Trades:** {ds['count']} &nbsp;&nbsp; "
            f"**Skewness:** {ds['skewness']:.2f} &nbsp;&nbsp; "
            f"**Kurtosis:** {ds['kurtosis']:.2f} &nbsp;&nbsp; "
            f"**Normal:** <span style='color:{_norm_clr}'>{_norm_txt}</span>",
            unsafe_allow_html=True,
        )

        if len(pnl_vals) > 0:
            # Create bins
            n_bins = min(40, max(10, len(pnl_vals) // 10))
            counts, bin_edges = np.histogram(pnl_vals, bins=n_bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Colors: red for negative bins, green for positive, with intensity by count
            max_count = max(counts) if max(counts) > 0 else 1
            colors = []
            for center, count in zip(bin_centers, counts):
                intensity = 0.3 + 0.7 * (count / max_count)
                if center < 0:
                    colors.append(f"rgba(255,82,82,{intensity})")
                else:
                    colors.append(f"rgba(0,230,118,{intensity})")

            fig_dist = go.Figure(go.Bar(
                x=[f"${c:,.0f}" for c in bin_centers],
                y=counts,
                marker_color=colors,
                hovertemplate="Range: %{x}<br>Trades: %{y}<extra></extra>",
            ))
            fig_dist.update_layout(
                template="plotly_dark",
                xaxis_title="Trade P&L", yaxis_title="Trades",
                height=400, margin=dict(l=60, r=30, t=30, b=60),
            )
            # Add legend
            fig_dist.add_trace(go.Bar(x=[None], y=[None], marker_color="rgba(0,230,118,0.8)", name="Profitable bins", showlegend=True))
            fig_dist.add_trace(go.Bar(x=[None], y=[None], marker_color="rgba(255,82,82,0.8)", name="Loss bins", showlegend=True))
            fig_dist.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
            st.plotly_chart(fig_dist, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # EFFICIENCY
    # ══════════════════════════════════════════════════════════════════════════
    with t_eff:
        es = efficiency_stats(trades)

        # KPI cards
        eff_cols = st.columns(4)
        _eff_metrics = [
            ("Edge Ratio", f"{es['edge_ratio']:.2f}", "Avg MFE / Avg MAE"),
            ("Win Capture", f"{es['win_capture_pct']:.0f}%", "P&L / MFE for winners"),
            ("R-Expectancy", f"{es['r_expectancy']:.2f}", "Avg P&L / MAE"),
            ("Efficiency Score", f"{es['efficiency_score']}", "Composite 0-100"),
        ]
        for col, (label, val, desc) in zip(eff_cols, _eff_metrics):
            with col:
                st.markdown(
                    f"<div style='background:#1a1e2e;border-radius:8px;padding:16px;text-align:left;'>"
                    f"<div style='font-size:13px;color:#8892b0'>{label}</div>"
                    f"<div style='font-size:28px;font-weight:bold;color:#00e676'>{val}</div>"
                    f"<div style='font-size:11px;color:#5a6380'>{desc}</div></div>",
                    unsafe_allow_html=True,
                )

        # MFE vs MAE scatter
        if es["mfe_values"]:
            mfe = np.array(es["mfe_values"])
            mae = np.array(es["mae_values"])
            is_win = np.array(es["is_winner"])

            fig_eff = go.Figure()
            # Losers
            fig_eff.add_trace(go.Scatter(
                x=mae[~is_win], y=mfe[~is_win], mode="markers",
                marker=dict(color="rgba(255,82,82,0.7)", size=8),
                name="Losers",
                hovertemplate="MAE: $%{x:,.0f}<br>MFE: $%{y:,.0f}<extra>Loss</extra>",
            ))
            # Winners
            fig_eff.add_trace(go.Scatter(
                x=mae[is_win], y=mfe[is_win], mode="markers",
                marker=dict(color="rgba(41,121,255,0.7)", size=8),
                name="Winners",
                hovertemplate="MAE: $%{x:,.0f}<br>MFE: $%{y:,.0f}<extra>Win</extra>",
            ))
            # Diagonal line
            _max_val = max(mfe.max(), mae.max()) if len(mfe) > 0 else 100
            fig_eff.add_trace(go.Scatter(
                x=[0, _max_val], y=[0, _max_val], mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1),
                showlegend=False,
            ))
            fig_eff.update_layout(
                template="plotly_dark",
                xaxis_title="MAE (Adverse Excursion)",
                yaxis_title="MFE (Favorable Excursion)",
                height=450, margin=dict(l=60, r=30, t=30, b=60),
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_eff, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TIME
    # ══════════════════════════════════════════════════════════════════════════
    with t_time:
        ta = time_analysis(trades)

        time_view = st.radio("Group by", ["By Hour", "By Day", "By Month"], horizontal=True, key="an_time_view")

        if time_view == "By Hour":
            data = ta["by_hour"]
            keys = sorted(data.keys())
            labels = [f"{h}:00" for h in keys]
            best_label = f"Best Hour: **{ta.get('best_hour', '—')}:00**"
            worst_label = f"Worst Hour: **{ta.get('worst_hour', '—')}:00**"
        elif time_view == "By Day":
            data = ta["by_day"]
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            keys = [d for d in day_order if d in data]
            labels = [d[:3] for d in keys]
            best_label = f"Best Day: **{ta.get('best_day', '—')}**"
            worst_label = f"Worst Day: **{ta.get('worst_day', '—')}**"
        else:
            data = ta["by_month"]
            month_order = ["January","February","March","April","May","June",
                           "July","August","September","October","November","December"]
            keys = [m for m in month_order if m in data]
            labels = [m[:3] for m in keys]
            best_label = ""; worst_label = ""

        if keys:
            avg_pnls = [data[k]["avg_pnl"] for k in keys]
            counts = [data[k]["count"] for k in keys]
            win_rates = [data[k]["win_rate"] for k in keys]

            st.markdown(f"{best_label} &nbsp;&nbsp; {worst_label} &nbsp;&nbsp; "
                        f"**Holding vs P&L Correlation:** {np.corrcoef(range(len(avg_pnls)), avg_pnls)[0,1]*100:.0f}%"
                        if len(avg_pnls) > 1 else "", unsafe_allow_html=True)

            colors = ["#00e676" if v >= 0 else "#ff5252" for v in avg_pnls]

            fig_time = go.Figure(go.Bar(
                x=labels, y=avg_pnls, marker_color=colors,
                hovertemplate="%{x}: %{customdata[0]} trades (%{customdata[1]:.0f}% win rate)<br>Avg P&L: $%{y:,.2f}<extra></extra>",
                customdata=list(zip(counts, [wr*100 for wr in win_rates])),
            ))
            fig_time.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_time.update_layout(
                template="plotly_dark",
                yaxis_title="Avg P&L ($)", height=400,
                margin=dict(l=60, r=30, t=30, b=60),
            )
            st.plotly_chart(fig_time, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    with t_mc:
        mc_sims = st.selectbox("Simulations", [500, 1000, 2500, 5000], index=1, key="an_mc_sims")

        if st.button("Re-run Simulation", key="an_mc_run"):
            with st.spinner(f"Running {mc_sims:,} Monte Carlo simulations..."):
                mc = monte_carlo(trades, an_capital, n_simulations=mc_sims)
                st.session_state["an_mc"] = mc
        elif "an_mc" not in st.session_state:
            with st.spinner("Running 1,000 Monte Carlo simulations..."):
                mc = monte_carlo(trades, an_capital, n_simulations=1000)
                st.session_state["an_mc"] = mc

        if "an_mc" in st.session_state:
            mc = st.session_state["an_mc"]

            st.caption(f"Last run: {mc_sims:,} simulations")

            # Probability KPIs
            mc_cols = st.columns(4)
            _mc_metrics = [
                ("Prob. of Profit", f"{mc['prob_profit']:.1f}%",
                 "green" if mc['prob_profit'] > 50 else "red"),
                ("Prob. 2x Capital", f"{mc['prob_2x']:.1f}%",
                 "green" if mc['prob_2x'] > 10 else "yellow"),
                ("Prob. 50% DD", f"{mc['prob_50_dd']:.1f}%",
                 "green" if mc['prob_50_dd'] < 10 else "red"),
                ("Median Max DD", f"{mc['median_max_dd_pct']:.1f}%",
                 "green" if mc['median_max_dd_pct'] < 20 else "red"),
            ]
            for col, (label, val, clr) in zip(mc_cols, _mc_metrics):
                with col:
                    st.markdown(
                        f"<div style='background:#1a1e2e;border-radius:8px;padding:16px;text-align:center;'>"
                        f"<div style='font-size:12px;color:#8892b0'>{label}</div>"
                        f"<div style='font-size:24px;font-weight:bold;color:{clr}'>{val}</div></div>",
                        unsafe_allow_html=True,
                    )

            # Confidence interval
            ci = mc["confidence_95"]
            st.markdown(
                f"<div style='background:#1a1e2e;border-radius:8px;padding:16px;text-align:center;margin:12px 0;'>"
                f"<div style='font-size:13px;color:#8892b0'>95% Confidence Interval (Final Equity)</div>"
                f"<div style='font-size:20px;'>"
                f"<span style='color:#ff5252'>${ci[0]:,.2f}</span> &nbsp;to&nbsp; "
                f"<span style='color:#00e676'>${ci[1]:,.2f}</span></div>"
                f"<div style='font-size:12px;color:#5a6380'>Expected: ${mc['expected_final']:,.2f}</div></div>",
                unsafe_allow_html=True,
            )

            # Equity paths chart
            pcts = mc["percentiles"]
            n_trades = len(pcts["50"])
            x_axis = list(range(n_trades))

            fig_mc = go.Figure()

            # Sample paths (gray)
            paths = mc["paths"]
            sample_n = min(50, len(paths))
            for i in range(sample_n):
                fig_mc.add_trace(go.Scatter(
                    x=x_axis, y=paths[i], mode="lines",
                    line=dict(color="rgba(255,255,255,0.05)", width=0.5),
                    showlegend=False, hoverinfo="skip",
                ))

            # Percentile lines
            pct_colors = {"5": "#ff5252", "25": "#ff9800", "50": "#2979ff", "75": "#00e676", "95": "#00e676"}
            pct_names = {"5": "5% (Worst)", "25": "25%", "50": "Median", "75": "75%", "95": "95% (Best)"}
            for pk in ["5", "25", "50", "75", "95"]:
                fig_mc.add_trace(go.Scatter(
                    x=x_axis, y=pcts[pk], mode="lines",
                    line=dict(color=pct_colors[pk], width=2, dash="dash" if pk != "50" else "solid"),
                    name=pct_names[pk],
                ))

            fig_mc.add_hline(y=an_capital, line_dash="dash", line_color="rgba(255,255,255,0.3)")
            fig_mc.update_layout(
                template="plotly_dark", height=450,
                xaxis_title="Trade #", yaxis_title="Equity ($)",
                yaxis=dict(tickformat="$,.0f"),
                margin=dict(l=60, r=30, t=30, b=60),
                legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            )
            fig_mc.update_layout(
                annotations=[dict(
                    text=f"Simulated from {mc_sims:,} random reshuffles. Gray lines show 50 sample paths, colored lines show percentiles.",
                    xref="paper", yref="paper", x=0.5, y=-0.22, showarrow=False,
                    font=dict(size=11, color="#5a6380"),
                )]
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            # Stress Tests
            st.markdown("#### Stress Test Scenarios")
            stress = mc["stress_tests"]
            st_cols = st.columns(4)
            _stress_items = [
                ("Double Losses", "2x all losing trades", stress.get("double_losses", {})),
                ("Half Wins", "0.5x all winning trades", stress.get("half_wins", {})),
                ("Extended DD", "Worst 5 trades repeated", stress.get("extended_dd", {})),
                ("Reduced Win Rate", "-10% win probability", stress.get("reduced_wr", {})),
            ]
            for col, (title, desc, data) in zip(st_cols, _stress_items):
                with col:
                    ret = data.get("return_pct", 0)
                    dd = data.get("max_dd_pct", 0)
                    pp = data.get("profit_pct", 0)
                    ret_c = "green" if ret > 0 else "red"
                    st.markdown(
                        f"<div style='background:#1a1e2e;border-radius:8px;padding:14px;'>"
                        f"<div style='font-weight:bold;font-size:14px'>{title}</div>"
                        f"<div style='font-size:11px;color:#5a6380;margin-bottom:8px'>{desc}</div>"
                        f"<div style='display:flex;justify-content:space-between'><span>Return:</span>"
                        f"<span style='color:{ret_c}'>{ret:+.1f}%</span></div>"
                        f"<div style='display:flex;justify-content:space-between'><span>Max DD:</span>"
                        f"<span>{dd:.1f}%</span></div>"
                        f"<div style='display:flex;justify-content:space-between'><span>Profit %:</span>"
                        f"<span>{pp:.0f}%</span></div></div>",
                        unsafe_allow_html=True,
                    )

    # ══════════════════════════════════════════════════════════════════════════
    # PROP FIRM
    # ══════════════════════════════════════════════════════════════════════════
    with t_prop:
        st.markdown("#### Prop Firm Challenge Simulator")
        st.caption("Test whether your strategy can pass a prop firm evaluation.")

        pf_col1, pf_col2 = st.columns(2)
        with pf_col1:
            pf_account = st.number_input("Account Size ($)", value=50000, step=5000, key="pf_acct")
            pf_target = st.number_input("Profit Target ($)", value=3000, step=500, key="pf_target")
            pf_daily_loss = st.number_input("Max Daily Loss Limit ($)", value=2000, step=250, key="pf_daily")
        with pf_col2:
            pf_max_dd = st.number_input("Max Total Drawdown ($)", value=3000, step=500, key="pf_maxdd")
            pf_time = st.number_input("Time Limit (days)", value=30, step=5, key="pf_time")
            pf_sims = st.selectbox("Simulations", [500, 1000, 2500, 5000], index=1, key="pf_sims")

        pf_scale = st.select_slider(
            "Position Sizing Scale", options=[0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
            value=1.0, key="pf_scale",
        )

        if st.button("Run Prop Firm Simulation", type="primary", key="pf_run"):
            with st.spinner(f"Running {pf_sims:,} prop firm simulations..."):
                pf = prop_firm_simulation(
                    trades, pf_account, pf_target, pf_daily_loss, pf_max_dd,
                    pf_time, n_simulations=pf_sims, position_scale=pf_scale,
                )
                st.session_state["an_pf"] = pf

        if "an_pf" in st.session_state:
            pf = st.session_state["an_pf"]

            # Pass/Fail bar
            pass_rate = pf["pass_rate"]
            fail_pcts = pf["fail_reason_pcts"]

            # Stacked bar visual
            _pass_pct = pass_rate
            _fail_pct = 100 - pass_rate
            st.markdown(
                f"<div style='display:flex;height:30px;border-radius:6px;overflow:hidden;margin:10px 0;'>"
                f"<div style='width:{_pass_pct}%;background:#00e676;display:flex;align-items:center;justify-content:center;"
                f"font-weight:bold;color:#000;font-size:13px'>{_pass_pct:.0f}%</div>"
                f"<div style='width:{_fail_pct}%;background:#ff9800;display:flex;align-items:center;justify-content:center;"
                f"font-weight:bold;color:#000;font-size:13px'>{_fail_pct:.0f}%</div></div>",
                unsafe_allow_html=True,
            )

            pf_res_cols = st.columns(2)
            with pf_res_cols[0]:
                st.markdown(f"**Passed** {pass_rate:.1f}%")
                st.caption("Hit profit target before time ran out")
            with pf_res_cols[1]:
                # Show top fail reason
                _top_reason = max(fail_pcts, key=fail_pcts.get) if fail_pcts else "—"
                _reason_labels = {"daily_loss": "Daily Loss", "max_dd": "Max Drawdown", "time_expired": "Time Expired"}
                st.markdown(f"**Failed** {_fail_pct:.1f}%")
                st.caption(f"Top reason: {_reason_labels.get(_top_reason, _top_reason)} ({fail_pcts.get(_top_reason, 0):.1f}%)")

            # Fail breakdown
            st.markdown("")
            fb_cols = st.columns(3)
            _fb_items = [
                ("Daily Loss", fail_pcts.get("daily_loss", 0)),
                ("Max Drawdown", fail_pcts.get("max_dd", 0)),
                ("Time Expired", fail_pcts.get("time_expired", 0)),
            ]
            for col, (label, pct) in zip(fb_cols, _fb_items):
                with col:
                    st.metric(label, f"{pct:.1f}%")

            # Simulated Equity Paths
            st.markdown("#### Simulated Equity Paths")
            st.caption(
                f"Shows how account equity evolves over {pf_time} days across all simulations. "
                "The blue line is the median outcome. Green/red bands show the range of possibilities."
            )

            pf_pcts = pf.get("percentiles", {})
            pf_paths = pf.get("equity_paths", np.array([]))

            if len(pf_pcts) > 0 and "50" in pf_pcts:
                n_days = len(pf_pcts["50"])
                x_days = list(range(n_days))

                fig_pf = go.Figure()

                # Sample paths
                if len(pf_paths) > 0:
                    for i in range(min(50, len(pf_paths))):
                        fig_pf.add_trace(go.Scatter(
                            x=x_days, y=pf_paths[i], mode="lines",
                            line=dict(color="rgba(255,255,255,0.04)", width=0.5),
                            showlegend=False, hoverinfo="skip",
                        ))

                # Percentile bands
                for pk, clr, nm in [("5","#ff5252","Worst 5%"),("25","#ff9800","25th pct"),
                                     ("50","#2979ff","Median"),("75","#00e676","75th pct"),("95","#00e676","Best 5%")]:
                    if pk in pf_pcts:
                        fig_pf.add_trace(go.Scatter(
                            x=x_days, y=pf_pcts[pk], mode="lines",
                            line=dict(color=clr, width=2, dash="dash" if pk != "50" else "solid"),
                            name=nm,
                        ))

                # Target line
                fig_pf.add_hline(y=pf_account + pf_target, line_dash="dash", line_color="#00e676",
                                 opacity=0.5, annotation_text="Target")
                fig_pf.add_hline(y=pf_account, line_dash="dash", line_color="rgba(255,255,255,0.3)")

                fig_pf.update_layout(
                    template="plotly_dark", height=400,
                    xaxis_title="Day", yaxis_title="Equity ($)",
                    yaxis=dict(tickformat="$,.0f"),
                    margin=dict(l=60, r=30, t=30, b=60),
                    legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                )
                st.plotly_chart(fig_pf, use_container_width=True)

            # Historical Timeline
            hist_results = pf.get("historical_results", [])
            hist_pass_rate = pf.get("historical_pass_rate", 0)
            if hist_results:
                st.markdown(f"#### Historical Timeline")
                st.caption(f'Each square = "if you started the challenge on this date, would you pass?" '
                           f'Green = passed ({sum(1 for r in hist_results if r["result"]=="passed")} of {len(hist_results)})')

                # Build dot grid
                _dot_colors = {
                    "passed": "#00e676", "daily_loss": "#ff5252",
                    "max_dd": "#ff9800", "time_expired": "#ffd740",
                }
                dots_html = ""
                for r in hist_results:
                    c = _dot_colors.get(r["result"], "#555")
                    _sd = r["start_date"]
                    _rs = r["result"]
                    dots_html += f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{c};margin:2px;' title='{_sd}: {_rs}'></span>"

                st.markdown(f"<div style='line-height:18px;'>{dots_html}</div>", unsafe_allow_html=True)

                # Legend
                st.markdown(
                    "<div style='margin-top:8px;font-size:12px;color:#8892b0'>"
                    "<span style='color:#00e676'>&#9679;</span> Passed &nbsp;&nbsp;"
                    "<span style='color:#ff5252'>&#9679;</span> Daily Loss &nbsp;&nbsp;"
                    "<span style='color:#ff9800'>&#9679;</span> Max DD &nbsp;&nbsp;"
                    "<span style='color:#ffd740'>&#9679;</span> Time Expired</div>",
                    unsafe_allow_html=True,
                )

    # ══════════════════════════════════════════════════════════════════════════
    # TRADES (raw list)
    # ══════════════════════════════════════════════════════════════════════════
    with t_trades:
        if trades:
            trades_df = pd.DataFrame(trades)
            display_df = pd.DataFrame()
            display_df["Date"] = pd.to_datetime(trades_df["entry_time"]).dt.strftime("%Y-%m-%d")
            display_df["Entry Time"] = pd.to_datetime(trades_df["entry_time"]).dt.strftime("%H:%M")
            display_df["Exit Time"] = pd.to_datetime(trades_df["exit_time"]).dt.strftime("%H:%M")
            display_df["Signal"] = trades_df["direction"].map({1: "LONG", -1: "SHORT"})
            display_df["Entry Price"] = trades_df["entry_price"].round(2)
            display_df["Exit Price"] = trades_df["exit_price"].round(2)
            display_df["Contracts"] = trades_df["contracts"]
            display_df["Entry Label"] = trades_df.get("entry_type", "—")
            display_df["Exit Label"] = trades_df.get("exit_reason", "—")
            display_df["P&L"] = trades_df["pnl"].round(2)

            def _style_pnl(val):
                if val > 0:
                    return "color: #00e676; font-weight: bold"
                elif val < 0:
                    return "color: #ff5252; font-weight: bold"
                return "color: #888"

            def _style_signal(val):
                return "color: #00e676" if val == "LONG" else ("color: #ff5252" if val == "SHORT" else "")

            styled = display_df.style.map(
                _style_pnl, subset=["P&L"]
            ).map(
                _style_signal, subset=["Signal"]
            ).format({
                "Entry Price": "{:.2f}",
                "Exit Price": "{:.2f}",
                "P&L": "${:,.2f}",
            })

            st.dataframe(styled, use_container_width=True, height=500)

            st.download_button(
                "Download Trade List (CSV)",
                data=display_df.to_csv(index=False),
                file_name=f"{_anr['strategy_key']}_analysis_trades.csv",
                mime="text/csv",
                key="an_dl_trades",
            )
        else:
            st.info("No trades to display.")
