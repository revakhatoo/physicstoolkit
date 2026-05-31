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

/* 3. Strip text from right-side buttons just in case to prevent duplicates */
[data-testid="stHeaderActionElements"] button::after {
    content: none !important;
}

/* 4. Prevent clipping on the Settings button and inject text */
[data-testid="stHeaderActionElements"] button:last-child {
    width: auto !important;
    padding-right: 10px !important;
    overflow: visible !important;
}
[data-testid="stHeaderActionElements"] button:last-child::after {
    content: " Settings" !important;
    margin-left: 5px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    vertical-align: middle !important;
}

/* 5. Force scrollbars to be thicker and always visible on tables */
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
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
        st.markdown("### Input Readings")
        
        # Default data always stays generic for input
        default_data = pd.DataFrame({"Measurement (x)": [10.00, 20.50, 35.67, 27.30]})
        edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
        
        unit_input = st.text_input("Unit (optional)", "")
        
        # Clean data so the UI doesn't vanish while editing
        clean_df = edited_df.dropna()
        
        if not clean_df.empty and len(clean_df) > 1:
            mean_val = clean_df["Measurement (x)"].mean()
            std_dev = clean_df["Measurement (x)"].std(ddof=1)
            std_error = std_dev / np.sqrt(len(clean_df))
            
            st.success(f"**Mean:** {mean_val:.2f}")
            st.info(f"**Std Deviation ($\\sigma$):** {std_dev:.2f}")
            st.info(f"**Standard Error ($\\pm$):** {std_error:.2f}")
        else:
            st.warning("Please enter at least two data points.")
            mean_val, std_dev, std_error = 0, 0, 0
        
    with display_col:
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
            
            st.dataframe(analysis_df.style.format("{:.2f}"), use_container_width=True)
            
            # --- Generate Matplotlib Table for PNG/PDF Exports ---
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
                st.download_button(
                    label="📥 Download CSV",
                    data=analysis_df.to_csv(index=False).encode('utf-8'),
                    file_name='error_analysis.csv',
                    mime='text/csv',
                )
            with dl_col2:
                st.download_button(
                    label="📥 Download PNG",
                    data=buf_tbl_png.getvalue(),
                    file_name='error_analysis_table.png',
                    mime='image/png',
                )
            with dl_col3:
                st.download_button(
                    label="📥 Download PDF",
                    data=buf_tbl_pdf.getvalue(),
                    file_name='error_analysis_table.pdf',
                    mime='application/pdf',
                )
            
            st.markdown("---")
            
            st.markdown("##### Standard Deviation ($\\sigma$)")
            st.latex(r"\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} \approx " + f"{std_dev:.2f}")
            with st.expander("Show Raw LaTeX for Standard Deviation"):
                st.code(r"\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}", language="latex")

            st.markdown("##### Standard Error (SE)")
            st.latex(r"SE = \frac{\sigma}{\sqrt{n}} \approx " + f"{std_error:.2f}")
            with st.expander("Show Raw LaTeX for Standard Error"):
                st.code(r"SE = \frac{\sigma}{\sqrt{n}}", language="latex")
            
            st.markdown("##### Final Reported Result:")
            val_str = f"{mean_val:.2f}"
            unc_str = f"{std_error:.2f}"
            unit_str = f" {unit_input}" if unit_input else ""
            latex_unit = f" \\text{{ {unit_input}}}" if unit_input else ""
            
            st.success(f"{val_str} ± {unc_str}{unit_str}")
            with st.expander("Show Raw LaTeX for Final Result"):
                st.code(f"{val_str} \\pm {unc_str}{latex_unit}", language="latex")

# ==========================================
# TOOL 2: GRAPHICAL ANALYSIS
# ==========================================
elif tool == "Graphical Analysis":
    st.title("Linear Regression & Best Fit")
    st.markdown("Enter your independent (X) and dependent (Y) variables to generate a line of best fit and extract experimental constants.")
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
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
            if len(x) > 1 and len(x) == len(y):
                coeffs, cov = np.polyfit(x, y, 1, cov=True)
                slope, intercept = coeffs
                slope_err = np.sqrt(cov[0][0])
                intercept_err = np.sqrt(cov[1][1])
                
                correlation_matrix = np.corrcoef(x, y)
                r_squared = correlation_matrix[0,1]**2
                
                m_str, m_err_str = format_sig_figs(slope, slope_err)
                c_str, c_err_str = format_sig_figs(intercept, intercept_err)
                
                st.info(f"**Slope (m):** {m_str} ± {m_err_str}")
                st.info(f"**Intercept (c):** {c_str} ± {c_err_str}")
                st.success(f"**R² Value:** {r_squared:.4f}")
                
                st.latex(r"y = mx + c")
                st.latex(f"y = ({m_str})x + ({c_str})")
            else:
                st.info("Awaiting 2+ complete data points for regression...")
                
        with plot_col:
            st.markdown("#### Scatter Plot")
            fig, ax = plt.subplots()
            
            if len(x) > 0:
                ax.scatter(x, y, label=data_point_label, color="#3182ce")
            
            if len(x) > 1 and len(x) == len(y):
                ax.plot(x, slope*x + intercept, color="#e53e3e", linestyle="--", label=best_fit_label)
                
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            if len(x) > 0:
                ax.legend()
                
            ax.grid(True, linestyle=":", alpha=0.7)
            st.pyplot(fig)
            
            if len(x) > 1 and len(x) == len(y):
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