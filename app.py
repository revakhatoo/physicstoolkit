import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import io

# Connect to your backend brain
from engine import get_final_unit, format_sig_figs, propagate_uncertainty, run_monte_carlo

# 1. Lock the sidebar to always open
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_title="Uncertainty Analysis")

# --- CUSTOM CSS FOR MOBILE UX ---
st.markdown("""
<style>
/* 1. Prevent Streamlit from clipping the injected text by forcing the left button to expand */
[data-testid="collapsedControl"] {
    width: auto !important;
    padding-right: 15px !important;
    overflow: visible !important;
}

/* 2. Inject "Navigation" next to the left arrows */
[data-testid="collapsedControl"]::after {
    content: " Navigation" !important;
    margin-left: 5px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    vertical-align: middle !important;
}

/* 3. STRICTLY hide any injected text from right-side header elements */
[data-testid="stHeaderActionElements"] *,
[data-testid="stHeaderActionElements"] *::before,
[data-testid="stHeaderActionElements"] *::after {
    content: none !important;
}

/* 4. Force scrollbars to be thicker and always visible on tables */
::-webkit-scrollbar {
    height: 12px !important;
}
::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05); 
    border-radius: 6px;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2); 
    border-radius: 6px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.4); 
}
</style>
""", unsafe_allow_html=True)

# 2. Setup the Navigation in the sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Select Tool:")
tool = st.sidebar.radio("Navigation", ["Reading based", "Graphical Analysis", "Formula based"], label_visibility="collapsed")

# ==========================================
# TOOL 1: READING BASED
# ==========================================
if tool == "Reading based":
    st.title("Error Analysis")
    st.markdown("Calculate statistical (Type A) and instrumental (Type B) uncertainties.")
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
        st.markdown("### 1. Measurement Method")
        lc_mode = st.radio("Select Method", ["Without Least Count (Simple)", "With Least Count (Vernier / Screw Gauge)"], label_visibility="collapsed")
        
        st.markdown("### 2. Input Readings")
        
        if lc_mode == "Without Least Count (Simple)":
            default_data = pd.DataFrame({"Measurement (x)": [10.00, 20.50, 35.67, 27.30]})
            edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            unit_input = st.text_input("Unit (optional)", "")
            
            clean_df = edited_df.dropna()
            
            if not clean_df.empty and len(clean_df) > 1:
                mean_val = clean_df["Measurement (x)"].mean()
                std_dev = clean_df["Measurement (x)"].std(ddof=1)
                std_error = std_dev / np.sqrt(len(clean_df))
                
                st.success(f"**Mean:** {mean_val:.4g}")
                st.info(f"**Std Deviation ($\\sigma$):** {std_dev:.4g}")
                st.info(f"**Standard Error ($\\pm$):** {std_error:.4g}")
            else:
                st.warning("Please enter at least two data points.")
                mean_val, std_dev, std_error = 0, 0, 0
                
        else: # WITH LEAST COUNT
            # LC is now extracted from the table into a clean universal input box
            col_lc, col_unit = st.columns(2)
            least_count = col_lc.number_input("Least Count", value=0.01, min_value=0.0, format="%.4f")
            unit_input = col_unit.text_input("Unit (optional)", "cm")
            
            default_data = pd.DataFrame({
                "Main Scale Reading (MSR)": [2.4, 2.4, 2.4, 2.5],
                "Vernier Scale Reading (VSR)": [5.0, 6.0, 4.0, 1.0]
            })
            edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
            
            clean_df = edited_df.dropna()
            
            if not clean_df.empty and len(clean_df) > 0:
                # Calculate Final Reading for the preview
                temp_final = clean_df["Main Scale Reading (MSR)"] + (clean_df["Vernier Scale Reading (VSR)"] * least_count)
                mean_val = temp_final.mean()
                
                st.success(f"**Average Final Reading:** {mean_val:.2f}")
                st.info(f"**Least Count ($\\pm$):** {least_count:.4g}")
            else:
                st.warning("Please enter at least one data point.")
                mean_val = 0
        
    with display_col:
        if lc_mode == "Without Least Count (Simple)":
            if not clean_df.empty and len(clean_df) > 1:
                with st.expander("⚙️ Customize Table Headings"):
                    col1_name = st.text_input("Measurement Column", "Measurement (x)")
                    col2_name = st.text_input("Mean Column", "Mean (x̄)")
                    col3_name = st.text_input("Deviation Column", "Deviation (x_i - x̄)")
                    col4_name = st.text_input("Squared Deviation Column", "Squared Deviation ((x_i - x̄)²)")
                
                st.info("👉 **Mobile tip:** Swipe the table left and right to view all columns.")
                
                analysis_df = clean_df.copy()
                analysis_df.rename(columns={"Measurement (x)": col1_name}, inplace=True)
                analysis_df[col2_name] = mean_val
                analysis_df[col3_name] = analysis_df[col1_name] - mean_val
                analysis_df[col4_name] = analysis_df[col3_name]**2
                
                st.dataframe(analysis_df.style.format("{:.3f}"), use_container_width=True)
                
                # PNG/PDF Export logic
                fig_tbl, ax_tbl = plt.subplots(figsize=(10, len(analysis_df) * 0.5 + 1))
                ax_tbl.axis('off')
                ax_tbl.axis('tight')
                display_data = analysis_df.round(3).astype(str)
                tbl = ax_tbl.table(cellText=display_data.values, colLabels=display_data.columns, loc='center', cellLoc='center')
                tbl.scale(1, 1.5) 
                
                buf_tbl_png = io.BytesIO()
                fig_tbl.savefig(buf_tbl_png, format="png", bbox_inches="tight", dpi=300)
                buf_tbl_pdf = io.BytesIO()
                fig_tbl.savefig(buf_tbl_pdf, format="pdf", bbox_inches="tight")
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    st.download_button(label="📥 Download CSV", data=analysis_df.to_csv(index=False).encode('utf-8'), file_name='error_analysis.csv', mime='text/csv')
                with dl_col2:
                    st.download_button(label="📥 Download PNG", data=buf_tbl_png.getvalue(), file_name='error_analysis_table.png', mime='image/png')
                with dl_col3:
                    st.download_button(label="📥 Download PDF", data=buf_tbl_pdf.getvalue(), file_name='error_analysis_table.pdf', mime='application/pdf')
                
                st.markdown("---")
                
                st.markdown("##### Standard Deviation ($\\sigma$)")
                st.latex(r"\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} \approx " + f"{std_dev:.4g}")
                
                st.markdown("##### Standard Error (SE)")
                st.latex(r"SE = \frac{\sigma}{\sqrt{n}} \approx " + f"{std_error:.4g}")
                
                st.markdown("##### Final Reported Result:")
                val_str = f"{mean_val:.4g}"
                unc_str = f"{std_error:.4g}"
                unit_str = f" {unit_input}" if unit_input else ""
                latex_unit = f" \\text{{ {unit_input}}}" if unit_input else ""
                
                st.success(f"{val_str} ± {unc_str}{unit_str}")
                with st.expander("Show Raw LaTeX for Final Result"):
                    st.code(f"{val_str} \\pm {unc_str}{latex_unit}", language="latex")
                    
        else: # DISPLAY FOR LEAST COUNT MODE
            if not clean_df.empty and len(clean_df) > 0:
                st.info("👉 **Mobile tip:** Swipe the table left and right to view all columns.")
                
                analysis_df = clean_df.copy()
                # Automatically calculate the Final Reading column using the single Least Count value
                analysis_df["Final Reading"] = analysis_df["Main Scale Reading (MSR)"] + (analysis_df["Vernier Scale Reading (VSR)"] * least_count)
                
                # Format to exactly 2 decimal places as requested
                st.dataframe(analysis_df.style.format("{:.2f}"), use_container_width=True)
                
                # PNG/PDF Export logic
                fig_tbl, ax_tbl = plt.subplots(figsize=(10, len(analysis_df) * 0.5 + 1))
                ax_tbl.axis('off')
                ax_tbl.axis('tight')
                display_data = analysis_df.round(2).astype(str)
                tbl = ax_tbl.table(cellText=display_data.values, colLabels=display_data.columns, loc='center', cellLoc='center')
                tbl.scale(1, 1.5) 
                
                buf_tbl_png = io.BytesIO()
                fig_tbl.savefig(buf_tbl_png, format="png", bbox_inches="tight", dpi=300)
                buf_tbl_pdf = io.BytesIO()
                fig_tbl.savefig(buf_tbl_pdf, format="pdf", bbox_inches="tight")
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                with dl_col1:
                    st.download_button(label="📥 Download CSV", data=analysis_df.to_csv(index=False).encode('utf-8'), file_name='least_count_analysis.csv', mime='text/csv')
                with dl_col2:
                    st.download_button(label="📥 Download PNG", data=buf_tbl_png.getvalue(), file_name='least_count_analysis.png', mime='image/png')
                with dl_col3:
                    st.download_button(label="📥 Download PDF", data=buf_tbl_pdf.getvalue(), file_name='least_count_analysis.pdf', mime='application/pdf')
                
                st.markdown("---")
                
                st.markdown("##### Final Reading Formula")
                st.latex(r"\text{Final Reading} = \text{MSR} + (\text{VSR} \times \text{LC})")
                
                st.markdown("##### Final Reported Result:")
                st.markdown("*Note: The final uncertainty is defined by the instrument's Least Count limit.*")
                
                # Locked to 2 decimal places to match the table
                val_str = f"{mean_val:.2f}"
                unc_str = f"{least_count:.4g}"
                unit_str = f" {unit_input}" if unit_input else ""
                latex_unit = f" \\text{{ {unit_input}}}" if unit_input else ""
                
                st.success(f"{val_str} ± {unc_str}{unit_str}")
                with st.expander("Show Raw LaTeX for Final Result"):
                    st.code(f"{val_str} \\pm {unc_str}{latex_unit}", language="latex")

# ==========================================
# TOOL 2: GRAPHICAL ANALYSIS
# ==========================================
elif tool == "Graphical Analysis":
    st.title("Curve Fitting & Regression")
    st.markdown("Enter your independent (X) and dependent (Y) variables to generate a curve of best fit and extract experimental constants.")
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
        
        fit_type = st.selectbox("Fitting Function", [
            "Linear (y = mx + c)",
            "Quadratic (y = ax² + bx + c)",
            "Exponential (y = A e^{Bx})",
            "Power (y = A x^B)"
        ])
        
        st.markdown("### Data Points")
        graph_data = pd.DataFrame({
            "X Values": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Y Values": [2.1, 4.0, 6.2, 7.9, 10.1]
        })
        edited_graph_df = st.data_editor(graph_data, num_rows="dynamic", use_container_width=True)
        
        st.markdown("### Plot Labels & Legends")
        x_label = st.text_input("X-Axis Label", "X axis")
        y_label = st.text_input("Y-Axis Label", "Y axis")
        data_point_label = st.text_input("Data Points Legend", "Raw Data")
        best_fit_label = st.text_input("Best Fit Legend", "Best Fit")
        
    with display_col:
        metrics_col, plot_col = st.columns([1, 2])
        
        clean_graph_df = edited_graph_df.dropna()
        x = clean_graph_df["X Values"].values
        y = clean_graph_df["Y Values"].values
        
        with metrics_col:
            st.markdown("#### Regression Results")
            if len(x) > 2:
                x_line = np.linspace(min(x), max(x), 100)
                fit_successful = False
                
                try:
                    if "Linear" in fit_type:
                        coeffs, cov = np.polyfit(x, y, 1, cov=True)
                        m, c = coeffs
                        dm, dc = np.sqrt(np.diag(cov))
                        y_pred = m*x + c
                        y_line = m*x_line + c
                        
                        m_str, m_err_str = format_sig_figs(m, dm)
                        c_str, c_err_str = format_sig_figs(c, dc)
                        
                        st.info(f"**Slope (m):** {m_str} ± {m_err_str}")
                        st.info(f"**Intercept (c):** {c_str} ± {c_err_str}")
                        st.latex(f"y = ({m_str})x + ({c_str})")
                        fit_successful = True
                        
                    elif "Quadratic" in fit_type:
                        coeffs, cov = np.polyfit(x, y, 2, cov=True)
                        a, b, c = coeffs
                        da, db, dc = np.sqrt(np.diag(cov))
                        y_pred = a*x**2 + b*x + c
                        y_line = a*x_line**2 + b*x_line + c
                        
                        a_str, a_err_str = format_sig_figs(a, da)
                        b_str, b_err_str = format_sig_figs(b, db)
                        c_str, c_err_str = format_sig_figs(c, dc)
                        
                        st.info(f"**a (x²):** {a_str} ± {a_err_str}")
                        st.info(f"**b (x):** {b_str} ± {b_err_str}")
                        st.info(f"**c (const):** {c_str} ± {c_err_str}")
                        st.latex(f"y = ({a_str})x^2 + ({b_str})x + ({c_str})")
                        fit_successful = True
                        
                    elif "Exponential" in fit_type:
                        valid = y > 0
                        if np.sum(valid) > 2:
                            x_v, y_v = x[valid], y[valid]
                            coeffs, cov = np.polyfit(x_v, np.log(y_v), 1, cov=True)
                            B, lnA = coeffs
                            dB, dlnA = np.sqrt(np.diag(cov))
                            A = np.exp(lnA)
                            dA = A * dlnA 
                            
                            y_pred = A * np.exp(B * x)
                            y_line = A * np.exp(B * x_line)
                            
                            A_str, A_err_str = format_sig_figs(A, dA)
                            B_str, B_err_str = format_sig_figs(B, dB)
                            
                            st.info(f"**Coefficient (A):** {A_str} ± {A_err_str}")
                            st.info(f"**Exponent (B):** {B_str} ± {B_err_str}")
                            st.latex(f"y = ({A_str}) e^{{({B_str})x}}")
                            fit_successful = True
                        else:
                            st.error("Exponential fits require y values > 0.")
                            
                    elif "Power" in fit_type:
                        valid = (x > 0) & (y > 0)
                        if np.sum(valid) > 2:
                            x_v, y_v = x[valid], y[valid]
                            coeffs, cov = np.polyfit(np.log(x_v), np.log(y_v), 1, cov=True)
                            B, lnA = coeffs
                            dB, dlnA = np.sqrt(np.diag(cov))
                            A = np.exp(lnA)
                            dA = A * dlnA
                            
                            y_pred = A * (x**B)
                            y_line = A * (x_line**B)
                            
                            A_str, A_err_str = format_sig_figs(A, dA)
                            B_str, B_err_str = format_sig_figs(B, dB)
                            
                            st.info(f"**Coefficient (A):** {A_str} ± {A_err_str}")
                            st.info(f"**Power (B):** {B_str} ± {B_err_str}")
                            st.latex(f"y = ({A_str}) x^{{({B_str})}}")
                            fit_successful = True
                        else:
                            st.error("Power fits require x and y values > 0.")
                            
                except Exception as e:
                    st.error("Fit failed. Please check your data distribution.")
                    fit_successful = False

                if fit_successful:
                    ss_res = np.sum((y - y_pred)**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    st.success(f"**R² Value:** {r_squared:.4f}")
                    
            else:
                st.info("Awaiting 3+ valid data points to calculate fit covariance...")
                fit_successful = False
                
        with plot_col:
            st.markdown("#### Scatter Plot")
            fig, ax = plt.subplots()
            
            if len(x) > 0:
                ax.scatter(x, y, label=data_point_label, color="#3182ce")
            
            if fit_successful:
                ax.plot(x_line, y_line, color="#e53e3e", linestyle="--", label=best_fit_label)
                
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if len(x) > 0:
                ax.legend()
                
            ax.grid(True, linestyle=":", alpha=0.7)
            st.pyplot(fig)
            
            if fit_successful:
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format="png", bbox_inches="tight", dpi=300)
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
                
                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(label="📥 Download PNG", data=buf_png.getvalue(), file_name="graph.png", mime="image/png")
                dl_col2.download_button(label="📥 Download PDF", data=buf_pdf.getvalue(), file_name="graph.pdf", mime="application/pdf")

# ==========================================
# TOOL 3: FORMULA BASED
# ==========================================
elif tool == "Formula based":
    st.title("Uncertainty Analysis")
    st.markdown("Enter your formula and measurements below to evaluate error propagation.")
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
        
        st.markdown("### Calculation Method")
        calc_method = st.radio("Method", ["Linear Propagation (Taylor)", "Monte Carlo Simulation"], label_visibility="collapsed")
        
        st.markdown("### 1. Input Parameters")
        formula_input = st.text_input("Formula (e.g., d / t)", "d / t")
        
        with st.expander("💡 Formula Syntax Guide"):
            st.markdown("""
            **Basic Operations:**
            * Addition / Subtraction: `x + y` / `x - y`
            * Multiplication / Division: `x * y` / `x / y`
            * Powers: `x**2` *(Do not use x^2)*
            
            **Trigonometry & Logarithms:**
            * Sine / Cosine / Tangent: `sin(x)`, `cos(x)`, `tan(x)`
            * Natural Log (ln): `log(x)`
            * Log base 10: `log(x, 10)`
            * Exponential ($e^x$): `exp(x)`
            
            **Roots & Constants:**
            * Square Root: `sqrt(x)`
            * Pi ($\pi$): `pi`
            * Euler's number ($e$): `E`
            """)
        
        try:
            expr = sp.sympify(formula_input)
            symbols_list = [str(sym) for sym in expr.free_symbols]
            valid_formula = True
        except Exception:
            st.error("Invalid formula syntax.")
            symbols_list = []
            valid_formula = False
            
        variables = {}
        if valid_formula and symbols_list:
            st.markdown("### Variables")
            
            for sym in symbols_list:
                st.markdown(f"**Variable:** `{sym}`")
                
                col1, col2, col3 = st.columns(3)
                val = col1.number_input("Value", value=1.0, key=f"val_{sym}")
                unc = col2.number_input("Uncert (±)", value=0.1, min_value=0.0, format="%.4f", key=f"unc_{sym}")
                unit = col3.text_input("Unit", value="", key=f"unit_{sym}")
                
                variables[sym] = {'val': val, 'uncert': unc, 'unit': unit}
                st.markdown("---") 
                
        calc_button = st.button("Calculate", type="primary", use_container_width=True)

    with display_col:
        if valid_formula and symbols_list and calc_button:
            linear_val, linear_unc, step_data = propagate_uncertainty(formula_input, variables)
            final_unit = get_final_unit(formula_input, variables)
            
            if calc_method == "Linear Propagation (Taylor)":
                col_res, col_derive = st.columns([1, 1])
                val_str, unc_str = format_sig_figs(linear_val, linear_unc)
                
                with col_res:
                    st.markdown("### Final Result")
                    st.success(f"**Result:** {val_str} ± {unc_str} {final_unit}")
                    
                    with st.expander("Show Raw LaTeX (For Reports)"):
                        st.code(f"{val_str} \\pm {unc_str} \\text{{ {final_unit}}}", language="latex")
                        
                    st.markdown("### Uncertainty Contributions")
                    total_variance = sum([data['contribution'] for data in step_data.values()])
                    if total_variance > 0:
                        latex_contribs = []
                        for sym, data in step_data.items():
                            percentage = data['contribution'] / total_variance
                            st.markdown(f"**{sym} contribution: {percentage*100:.1f}%**")
                            st.progress(float(percentage))
                            latex_contribs.append(f"\\text{{{sym} contribution}} = {percentage*100:.1f}\\%")
                            
                        with st.expander("Show Raw LaTeX for Contributions"):
                            st.code(" \\\\\n".join(latex_contribs), language="latex")
                    
                with col_derive:
                    st.markdown("### Step-by-Step Derivation")
                    for sym, data in step_data.items():
                        st.markdown(f"Partial derivative with respect to **{sym}**:")
                        latex_deriv = sp.latex(data['symbolic'])
                        st.latex(f"\\frac{{\\partial}}{{\\partial {sym}}} = {latex_deriv}")
                        
                        st.markdown(f"<p style='text-align: center'>Derivative value = {data['evaluated']:.4g}</p>", unsafe_allow_html=True)
                        
                        with st.expander(f"Show Raw LaTeX for {sym} Derivative"):
                            st.code(f"\\frac{{\\partial f}}{{\\partial {sym}}} = {latex_deriv}", language="latex")
                            
                        st.markdown("---")
            
            elif calc_method == "Monte Carlo Simulation":
                mc_mean, mc_std, results, median, lower, upper = run_monte_carlo(formula_input, variables)
                
                if linear_val != 0 and linear_unc != 0:
                    val_diff = abs(linear_val - mc_mean) / abs(linear_val)
                    unc_diff = abs(linear_unc - mc_std) / linear_unc
                    if val_diff > 0.05 or unc_diff > 0.05:
                        st.warning("⚠️ **High Non-Linearity Detected:** The Monte Carlo standard deviation differs from the linear Taylor approximation. The formula may be highly sensitive to these uncertainties. Trust the Monte Carlo distribution.")
                
                mc_val_str, mc_unc_str = format_sig_figs(mc_mean, mc_std)
                
                st.markdown("### Monte Carlo Final Result (10,000 Iterations)")
                st.success(f"**Result:** {mc_val_str} ± {mc_unc_str} {final_unit}")
                
                with st.expander("Show Raw LaTeX (For Reports)"):
                    st.code(f"{mc_val_str} \\pm {mc_unc_str} \\text{{ {final_unit}}}", language="latex")
                
                st.markdown(f"**Median:** {median:.4g} | **Asymmetric Bounds:** +{upper:.4g} / -{lower:.4g}")
                
                fig_mc, ax_mc = plt.subplots()
                ax_mc.hist(results, bins=50, color="#3182ce", edgecolor="black", alpha=0.7)
                ax_mc.axvline(mc_mean, color="#e53e3e", linestyle="dashed", linewidth=2, label=f"Mean: {mc_mean:.4g}")
                ax_mc.set_xlabel(f"Calculated Result ({final_unit})")
                ax_mc.set_ylabel("Frequency")
                ax_mc.set_title("Monte Carlo Propagation Distribution")
                ax_mc.legend()
                ax_mc.grid(True, linestyle=":", alpha=0.7)
                
                st.pyplot(fig_mc)
                
                buf_mc_png = io.BytesIO()
                fig_mc.savefig(buf_mc_png, format="png", bbox_inches="tight", dpi=300)
                buf_mc_pdf = io.BytesIO()
                fig_mc.savefig(buf_mc_pdf, format="pdf", bbox_inches="tight")
                
                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(label="📥 Download Histogram (PNG)", data=buf_mc_png.getvalue(), file_name="mc_histogram.png", mime="image/png")
                dl_col2.download_button(label="📥 Download Histogram (PDF)", data=buf_mc_pdf.getvalue(), file_name="mc_histogram.pdf", mime="application/pdf")