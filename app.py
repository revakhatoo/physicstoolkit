import streamlit as st
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io 
from engine import propagate_uncertainty, run_monte_carlo, get_final_unit, format_sig_figs

st.set_page_config(page_title="Uncertainty Analysis", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION (AUTO-SAVE)
# ==========================================
if "type_a_data" not in st.session_state:
    st.session_state.type_a_data = pd.DataFrame({"Measurement (x)": [10.00, 20.50, 35.67, 27.30]})
if "graph_data" not in st.session_state:
    st.session_state.graph_data = pd.DataFrame({"X Values": [1.0, 2.0, 3.0, 4.0, 5.0], "Y Values": [2.1, 4.0, 6.2, 7.9, 10.1]})
if "formula_input" not in st.session_state:
    st.session_state.formula_input = "d / t"

# --- App Navigation Router ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Tool:", ["📊 Reading based", "📈 Graphical Analysis", "📐 Formula based"])
st.sidebar.divider()

# ==========================================
# MODE 1: RAW MEASUREMENTS (TYPE A)
# ==========================================
if app_mode == "📊 Reading based":
    st.title("Error Analysis")
    
    st.sidebar.header("Input Readings")
    st.session_state.type_a_data = st.sidebar.data_editor(st.session_state.type_a_data, num_rows="dynamic", hide_index=True)
    
    # Forces data to be numbers, turning accidental text into NaN, then drops them
    measurements = pd.to_numeric(st.session_state.type_a_data["Measurement (x)"], errors='coerce').dropna().to_numpy()
    valid_measurements = measurements[measurements != 0]
    
    if len(valid_measurements) > 1:
        n = len(valid_measurements)
        mean_val = np.round(np.mean(valid_measurements), 2)
        devs = np.round(valid_measurements - mean_val, 2)
        sq_devs = np.round(devs ** 2, 4)
        
        std_dev = np.sqrt(np.sum(sq_devs) / (n - 1))
        std_error = std_dev / np.sqrt(n)
        
        df_display = pd.DataFrame({
            "S.No.": range(1, n + 1),
            "Measurement (x)": [f"{x:.2f}" for x in valid_measurements],
            "Mean (x̄)": [f"{mean_val:.2f}"] * n,
            "Deviation (x_i - x̄)": [f"{d:.2f}" for d in devs],
            "Squared Deviation ((x_i - x̄)²)": [f"{sq:.2f}" for sq in sq_devs]
        })
        
        st.table(df_display)
        
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Table as CSV",
            data=csv,
            file_name='error_analysis_table.csv',
            mime='text/csv',
        )
        
        # --- RAW LATEX: Type A Equation ---
        latex_se = rf"\text{{Standard Error (SE)}} = \frac{{\sigma}}{{\sqrt{{n}}}} = \frac{{\sqrt{{\frac{{\sum (x_i - \bar{{x}})^2}}{{n-1}}}}}}{{\sqrt{{{n}}}}} \approx {std_error:.2f}"
        st.latex(latex_se)
        with st.expander("Show Raw LaTeX (For Reports)"):
            st.code(latex_se, language="latex")
        
        st.markdown("**Final Reported Result:**")
        st.success(f"{mean_val:.2f} ± {std_error:.2f}")
        
        # --- RAW LATEX: Type A Final Result ---
        latex_se_final = rf"{mean_val:.2f} \pm {std_error:.2f}"
        with st.expander("Show Raw LaTeX (For Reports)"):
            st.code(latex_se_final, language="latex")
        
        st.sidebar.divider()
        st.sidebar.success(f"**Mean:** {mean_val:.2f}")
        st.sidebar.info(f"**Standard Error (±):** {std_error:.2f}")
        st.sidebar.caption("Copy this Standard Error into the Formula tab.")
        
    elif len(valid_measurements) == 1:
        st.warning("Please enter at least 2 readings to calculate uncertainty.")

# ==========================================
# MODE 2: GRAPHICAL ANALYSIS (BEST FIT)
# ==========================================
elif app_mode == "📈 Graphical Analysis":
    st.title("Linear Regression & Best Fit")
    st.write("Enter your independent (X) and dependent (Y) variables to generate a line of best fit and extract experimental constants.")

    st.sidebar.header("Data Points")
    st.session_state.graph_data = st.sidebar.data_editor(st.session_state.graph_data, num_rows="dynamic", hide_index=True)
    
    x_vals = pd.to_numeric(st.session_state.graph_data["X Values"], errors='coerce').dropna().to_numpy()
    y_vals = pd.to_numeric(st.session_state.graph_data["Y Values"], errors='coerce').dropna().to_numpy()
    
    st.sidebar.divider()
    
    st.sidebar.header("Plot Labels & Legends")
    x_label = st.sidebar.text_input("X-Axis Label", value="x axis")
    y_label = st.sidebar.text_input("Y-Axis Label", value="y axis")
    
    data_legend = st.sidebar.text_input("Data point's label", value="Raw Data")
    fit_legend = st.sidebar.text_input("Line of best fit label", value="Best Fit")
    
    if len(x_vals) == len(y_vals) and len(x_vals) > 2:
        coeffs, cov = np.polyfit(x_vals, y_vals, 1, cov=True)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        slope_err = np.sqrt(cov[0, 0])
        intercept_err = np.sqrt(cov[1, 1])
        
        correlation_matrix = np.corrcoef(x_vals, y_vals)
        r_squared = correlation_matrix[0, 1] ** 2
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Regression Results")
            st.info(f"**Slope (m):** {slope:.4g} ± {slope_err:.4g}")
            st.info(f"**Intercept (c):** {intercept:.4g} ± {intercept_err:.4g}")
            
            if r_squared > 0.99:
                st.success(f"**R² Value:** {r_squared:.4f} (Excellent Fit)")
            elif r_squared > 0.95:
                st.warning(f"**R² Value:** {r_squared:.4f} (Good Fit)")
            else:
                st.error(f"**R² Value:** {r_squared:.4f} (Poor Fit)")
                
            # --- RAW LATEX: Regression Equation ---
            latex_gen_eq = r"y = mx + c"
            latex_sub_eq = rf"y = ({slope:.3g})x + ({intercept:.3g})"
            st.latex(latex_gen_eq)
            st.latex(latex_sub_eq)
            
            with st.expander("Show Raw LaTeX (For Reports)"):
                st.code(f"{latex_gen_eq}\n{latex_sub_eq}", language="latex")
            
        with col2:
            st.subheader("Scatter Plot")
            fig, ax = plt.subplots(figsize=(7, 6)) 
            
            ax.scatter(x_vals, y_vals, color='#378ADD', edgecolor='white', s=80, label=data_legend, zorder=5)
            
            x_max = np.max(x_vals)
            y_max = np.max(y_vals)
            
            x_limit = np.ceil(x_max + (x_max * 0.25))
            y_limit = np.ceil(y_max + (y_max * 0.25))
            
            x_line_max = x_max + (x_max * 0.05)
            x_line = np.linspace(0, x_line_max, 100)
            y_line = slope * x_line + intercept
            
            ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, label=fit_legend)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            ax.set_xlim(left=0, right=x_limit)
            ax.set_ylim(bottom=0, top=y_limit)
            
            x_tick_step = 1 if x_limit <= 15 else max(1, int(np.ceil(x_limit / 10)))
            y_tick_step = 1 if y_limit <= 15 else max(1, int(np.ceil(y_limit / 10)))
            
            ax.set_xticks(np.arange(0, x_limit + x_tick_step, x_tick_step))
            ax.set_yticks(np.arange(0, y_limit + y_tick_step, y_tick_step))
            
            ax.legend(loc='upper right', framealpha=0.95)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            fig.tight_layout() 
            st.pyplot(fig)
            
            st.markdown("##### Export Graph")
            col_png, col_pdf = st.columns(2)
            
            buf_png = io.BytesIO()
            fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
            buf_png.seek(0)
            
            buf_pdf = io.BytesIO()
            fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
            buf_pdf.seek(0)
            
            with col_png:
                st.download_button(label="📥 Download as PNG", data=buf_png, file_name="regression_graph.png", mime="image/png")
            with col_pdf:
                st.download_button(label="📥 Download as PDF", data=buf_pdf, file_name="regression_graph.pdf", mime="application/pdf")
            
    else:
        st.warning("Please ensure you have an equal number of X and Y values, and at least 3 data points to perform regression.")

# ==========================================
# MODE 3: FORMULA & VARIABLES (TYPE B)
# ==========================================
elif app_mode == "📐 Formula based":
    st.title("Uncertainty Analysis")
    st.write("Enter your formula and measurements below to see the step-by-step derivation.")

    st.sidebar.header("1. Input Parameters")
    
    formula = st.sidebar.text_input("Formula (e.g., d / t)", value=st.session_state.formula_input)
    st.session_state.formula_input = formula
    
    try:
        expr = sp.sympify(formula)
        st.sidebar.markdown("**Live Math Preview:**")
        st.sidebar.latex(sp.latex(expr))
    except Exception:
        pass
        
    with st.sidebar.expander("💡 Syntax Cheat Sheet"):
        st.markdown("""
        * **Power:** `x**2`
        * **Square Root:** `sqrt(x)`
        * **Trig:** `sin(x)`, `cos(x)`, `tan(x)`
        * **Exponential:** `exp(x)`
        * **Logarithm:** `log(x)`
        * **Pi:** `pi`
        """)
        
    st.sidebar.divider()
    variables = {}
    
    try:
        var_names = sorted([str(sym) for sym in expr.free_symbols])
        st.sidebar.subheader("Variables")
        
        for var in var_names:
            st.sidebar.markdown(f"**Variable: `{var}`**")
            
            val_key = f"val_{var}"
            unc_key = f"unc_{var}"
            unit_key = f"unit_{var}"
            
            default_val = st.session_state.get(val_key, 12.4 if var == 'd' else (3.1 if var == 't' else 1.0))
            default_unc = st.session_state.get(unc_key, 0.2 if var == 'd' else (0.05 if var == 't' else 0.1))
            default_unit = st.session_state.get(unit_key, "m" if var == 'd' else ("s" if var == 't' else "dimensionless"))
            
            col1, col2, col3 = st.sidebar.columns([2, 2, 2])
            with col1:
                val = st.number_input("Value", value=float(default_val), key=val_key)
            with col2:
                unc = st.number_input("Uncert (±)", value=float(default_unc), key=unc_key)
            with col3:
                unit = st.text_input("Unit", value=default_unit, key=unit_key)
                
            variables[var] = {'val': val, 'uncert': unc, 'unit': unit}
            st.sidebar.divider()
            
    except Exception:
        st.sidebar.error("Waiting for a valid formula...")

    st.sidebar.subheader("Method")
    method = st.sidebar.radio("Select propagation method:", ["Linear Propagation", "Monte Carlo Simulation"])

    if st.button("Calculate", type="primary", disabled=not variables):
        try:
            col1, col2 = st.columns(2)
            final_unit = get_final_unit(formula, variables)
            
            if method == "Linear Propagation":
                val, uncert, steps = propagate_uncertainty(formula, variables)
                fmt_val, fmt_unc = format_sig_figs(val, uncert)
                
                with col1:
                    st.subheader("Final Result")
                    if "Error" in final_unit:
                        st.error(final_unit)
                    else:
                        st.success(f"**Result:** {fmt_val} ± {fmt_unc} {final_unit}")
                        # --- RAW LATEX: Final Propagated Result ---
                        latex_final_ans = rf"{fmt_val} \pm {fmt_unc} \text{{ {final_unit}}}"
                        with st.expander("Show Raw LaTeX (For Reports)"):
                            st.code(latex_final_ans, language="latex")
                    
                    st.subheader("Uncertainty Contributions")
                    total_variance = sum(d['contribution'] for d in steps.values())
                    for var, data in steps.items():
                        pct = 0.0 if total_variance == 0 else (data['contribution'] / total_variance) * 100
                        st.write(f"**{var}** contribution: {pct:.1f}%")
                        st.progress(int(pct))

                with col2:
                    st.subheader("Step-by-Step Derivation")
                    for var, data in steps.items():
                        st.markdown(f"**Partial derivative with respect to `{var}`:**")
                        latex_deriv = sp.latex(data['symbolic'])
                        
                        latex_deriv_display = rf"\frac{{\partial}}{{\partial {var}}} = {latex_deriv}"
                        latex_eval_display = rf"\text{{Evaluated at }} {var} = {data['evaluated']:.4f}"
                        
                        st.latex(latex_deriv_display)
                        st.latex(latex_eval_display)
                        
                        with st.expander("Show Raw LaTeX (For Reports)"):
                            st.code(f"{latex_deriv_display}\n{latex_eval_display}", language="latex")
                            
                        st.divider()
                        
            elif method == "Monte Carlo Simulation":
                mc_mean, mc_std, results, median, lower_bound, upper_bound = run_monte_carlo(formula, variables, n_iterations=10000)
                fmt_mean, fmt_std = format_sig_figs(mc_mean, mc_std)
                
                with col1:
                    st.subheader("Final Result (Monte Carlo)")
                    if "Error" in final_unit:
                        st.error(final_unit)
                    else:
                        st.success(f"**Standard Result:** {fmt_mean} ± {fmt_std} {final_unit}")
                        # --- RAW LATEX: Final MC Result ---
                        latex_mc_ans = rf"{fmt_mean} \pm {fmt_std} \text{{ {final_unit}}}"
                        with st.expander("Show Raw LaTeX (For Reports)"):
                            st.code(latex_mc_ans, language="latex")
                        
                        asymmetry_ratio = abs(upper_bound - lower_bound) / (upper_bound + lower_bound) if (upper_bound + lower_bound) > 0 else 0
                        if asymmetry_ratio > 0.05:
                            st.warning("⚠️ **High Non-Linearity Detected**")
                            with st.expander("View Asymmetric Bounds (Professional)", expanded=True):
                                fmt_med, fmt_upper = format_sig_figs(median, upper_bound)
                                _, fmt_lower = format_sig_figs(median, lower_bound)
                                st.info(f"**Asymmetric Result:** {fmt_med}  (+{fmt_upper} / -{fmt_lower}) {final_unit}")
                                # --- RAW LATEX: Asymmetric bounds ---
                                latex_asym_ans = rf"{fmt_med}_{{-{fmt_lower}}}^{{+{fmt_upper}}} \text{{ {final_unit}}}"
                                st.code(latex_asym_ans, language="latex")
                        else:
                            st.info("Result is based on 10,000 simulated iterations drawing from Gaussian distributions.")
                
                with col2:
                    st.subheader("Output Distribution")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(results, bins=50, color='#378ADD', edgecolor='white', alpha=0.8)
                    ax.axvline(mc_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mc_mean:.2f}')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # --- NEW: Monte Carlo Graph Download Buttons ---
                    st.markdown("##### Export Distribution Graph")
                    col_png_mc, col_pdf_mc = st.columns(2)
                    
                    buf_png_mc = io.BytesIO()
                    fig.savefig(buf_png_mc, format="png", bbox_inches="tight", dpi=300)
                    buf_png_mc.seek(0)
                    
                    buf_pdf_mc = io.BytesIO()
                    fig.savefig(buf_pdf_mc, format="pdf", bbox_inches="tight")
                    buf_pdf_mc.seek(0)
                    
                    with col_png_mc:
                        st.download_button(label="📥 Download PNG", data=buf_png_mc, file_name="monte_carlo_distribution.png", mime="image/png")
                    with col_pdf_mc:
                        st.download_button(label="📥 Download PDF", data=buf_pdf_mc, file_name="monte_carlo_distribution.pdf", mime="application/pdf")
                    
        except Exception as e:
            st.error(f"Calculation Error: {e}")

# ==========================================
# GLOBAL SIDEBAR FOOTER
# ==========================================
st.sidebar.divider()
st.sidebar.info("💡 **Tip:** To save this entire page as a document, press `Ctrl+P` (or `Cmd+P` on Mac) and select **Save as PDF**.")