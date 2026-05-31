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
        # Default data
        default_data = pd.DataFrame({"Measurement (x)": [10.00, 20.50, 35.67, 27.30]})
        edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)
        
        # Clean data so the UI doesn't vanish while editing
        clean_df = edited_df.dropna()
        
        if not clean_df.empty and len(clean_df) > 1:
            mean_val = clean_df["Measurement (x)"].mean()
            std_dev = clean_df["Measurement (x)"].std(ddof=1)
            std_error = std_dev / np.sqrt(len(clean_df))
            
            st.success(f"**Mean ($\\bar{{x}}$):** {mean_val:.2f}")
            st.info(f"**Std Deviation ($\\sigma$):** {std_dev:.2f}")
            st.info(f"**Standard Error ($\\pm$):** {std_error:.2f}")
        else:
            st.warning("Please enter at least two data points.")
            mean_val, std_dev, std_error = 0, 0, 0
        
    with display_col:
        if not clean_df.empty and len(clean_df) > 1:
            analysis_df = clean_df.copy()
            analysis_df["Mean (x̄)"] = mean_val
            analysis_df["Deviation (x_i - x̄)"] = analysis_df["Measurement (x)"] - mean_val
            analysis_df["Squared Deviation ((x_i - x̄)²)"] = analysis_df["Deviation (x_i - x̄)"]**2
            
            st.dataframe(analysis_df.style.format("{:.2f}"), use_container_width=True)
            
            # --- NEW: Generate Matplotlib Table for PNG/PDF Exports ---
            # Create a figure sized dynamically based on the number of rows
            fig_tbl, ax_tbl = plt.subplots(figsize=(10, len(analysis_df) * 0.5 + 1))
            ax_tbl.axis('off')
            ax_tbl.axis('tight')
            
            # Format data to 2 decimal places for the image
            display_data = analysis_df.round(2).astype(str)
            tbl = ax_tbl.table(cellText=display_data.values, colLabels=display_data.columns, loc='center', cellLoc='center')
            tbl.scale(1, 1.5) # Add some padding to cells
            
            # Save to buffers
            buf_tbl_png = io.BytesIO()
            fig_tbl.savefig(buf_tbl_png, format="png", bbox_inches="tight", dpi=300)
            
            buf_tbl_pdf = io.BytesIO()
            fig_tbl.savefig(buf_tbl_pdf, format="pdf", bbox_inches="tight")
            
            # Display all three download options in a neat row
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
            
            # Show standard deviation explicitly
            st.markdown("##### Standard Deviation ($\\sigma$)")
            st.latex(r"\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} \approx " + f"{std_dev:.2f}")
            with st.expander("Show Raw LaTeX for Standard Deviation"):
                st.code(r"\sigma = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}", language="latex")

            st.markdown("##### Standard Error (SE)")
            st.latex(r"SE = \frac{\sigma}{\sqrt{n}} \approx " + f"{std_error:.2f}")
            with st.expander("Show Raw LaTeX for Standard Error"):
                st.code(r"SE = \frac{\sigma}{\sqrt{n}}", language="latex")
            
            st.markdown("##### Final Reported Result:")
            # Bypassing format_sig_figs to show exact UI matching decimals
            val_str = f"{mean_val:.2f}"
            unc_str = f"{std_error:.2f}"
            
            st.success(f"{val_str} ± {unc_str}")
            with st.expander("Show Raw LaTeX for Final Result"):
                st.code(f"{val_str} \\pm {unc_str}", language="latex")

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
        
        # Clean data but DO NOT hide the UI if it's empty
        clean_graph_df = edited_graph_df.dropna()
        x = clean_graph_df["X Values"].values
        y = clean_graph_df["Y Values"].values
        
        with metrics_col:
            st.markdown("#### Regression Results")
            if len(x) > 1 and len(x) == len(y):
                # Calculate simple regression
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
            
            # Always plot whatever points are valid
            if len(x) > 0:
                ax.scatter(x, y, label=data_point_label, color="#3182ce")
            
            # Only plot the line if we have enough points
            if len(x) > 1 and len(x) == len(y):
                ax.plot(x, slope*x + intercept, color="#e53e3e", linestyle="--", label=best_fit_label)
                
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            
            # Only show legend if we actually plotted something
            if len(x) > 0:
                ax.legend()
                
            ax.grid(True, linestyle=":", alpha=0.7)
            st.pyplot(fig)
            
            # Only allow downloads if a valid line exists
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
            
            **Calculus (Advanced):**
            * Derivative: `diff(x**2, x)`
            * Integral: `integrate(x**2, x)`
            """)
        
        # Dynamically extract variables
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
            
            # --- FIXED GRID ALIGNMENT ---
            h_sym, h1, h2, h3 = st.columns([0.5, 1, 1, 1])
            h_sym.markdown("**Var**")
            h1.markdown("**Value**")
            h2.markdown("**Uncert (±)**")
            h3.markdown("**Unit**")
            
            for sym in symbols_list:
                row_sym, row1, row2, row3 = st.columns([0.5, 1, 1, 1])
                
                row_sym.markdown(f"<div style='padding-top: 8px; font-weight: bold;'>{sym}</div>", unsafe_allow_html=True)
                
                val = row1.number_input(f"{sym} val", value=1.0, key=f"val_{sym}", label_visibility="collapsed")
                unc = row2.number_input(f"{sym} unc", value=0.1, min_value=0.0, format="%.4f", key=f"unc_{sym}", label_visibility="collapsed")
                unit = row3.text_input(f"{sym} unit", value="", key=f"unit_{sym}", label_visibility="collapsed")
                
                variables[sym] = {'val': val, 'uncert': unc, 'unit': unit}
                
        st.markdown("### Calculation Method")
        calc_method = st.radio("Method", ["Linear Propagation (Taylor)", "Monte Carlo Simulation"], label_visibility="collapsed")
        calc_button = st.button("Calculate", type="primary")

    with display_col:
        if valid_formula and symbols_list and calc_button:
            # We always run the linear derivation to get the step-by-step breakdown
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
                # Run the simulation
                mc_mean, mc_std, results, median, lower, upper = run_monte_carlo(formula_input, variables)
                
                # Nonlinearity Warning Check (Tolerance > 5%)
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
                
                # Plot Histogram
                fig_mc, ax_mc = plt.subplots()
                ax_mc.hist(results, bins=50, color="#3182ce", edgecolor="black", alpha=0.7)
                ax_mc.axvline(mc_mean, color="#e53e3e", linestyle="dashed", linewidth=2, label=f"Mean: {mc_mean:.4g}")
                ax_mc.set_xlabel(f"Calculated Result ({final_unit})")
                ax_mc.set_ylabel("Frequency")
                ax_mc.set_title("Monte Carlo Propagation Distribution")
                ax_mc.legend()
                ax_mc.grid(True, linestyle=":", alpha=0.7)
                
                st.pyplot(fig_mc)
                
                # Save histogram buffers
                buf_mc_png = io.BytesIO()
                fig_mc.savefig(buf_mc_png, format="png", bbox_inches="tight", dpi=300)
                buf_mc_pdf = io.BytesIO()
                fig_mc.savefig(buf_mc_pdf, format="pdf", bbox_inches="tight")
                
                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button(label="📥 Download Histogram (PNG)", data=buf_mc_png.getvalue(), file_name="mc_histogram.png", mime="image/png")
                dl_col2.download_button(label="📥 Download Histogram (PDF)", data=buf_mc_pdf.getvalue(), file_name="mc_histogram.pdf", mime="application/pdf")