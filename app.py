import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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
        
        # Calculate stats
        if not edited_df.empty and len(edited_df) > 1:
            mean_val = edited_df["Measurement (x)"].mean()
            std_error = edited_df["Measurement (x)"].std(ddof=1) / np.sqrt(len(edited_df))
            
            st.success(f"**Mean:** {mean_val:.2f}")
            st.info(f"**Standard Error (±):** {std_error:.2f}")
        else:
            st.warning("Please enter at least two data points.")
            mean_val, std_error = 0, 0
        
    with display_col:
        if not edited_df.empty and len(edited_df) > 1:
            analysis_df = edited_df.copy()
            analysis_df["Mean (x̄)"] = mean_val
            analysis_df["Deviation (x_i - x̄)"] = analysis_df["Measurement (x)"] - mean_val
            analysis_df["Squared Deviation ((x_i - x̄)²)"] = analysis_df["Deviation (x_i - x̄)"]**2
            
            st.dataframe(analysis_df.style.format("{:.2f}"), use_container_width=True)
            
            st.download_button(
                label="📥 Download Table as CSV",
                data=analysis_df.to_csv(index=False).encode('utf-8'),
                file_name='error_analysis.csv',
                mime='text/csv',
            )
            
            st.markdown("---")
            st.latex(r"Standard Error (SE) = \frac{\sigma}{\sqrt{n}} = \frac{\sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}}}{\sqrt{n}} \approx " + f"{std_error:.2f}")
            
            with st.expander("Show Raw LaTeX (For Reports)"):
                st.code(r"\text{Standard Error (SE)} = \frac{\sigma}{\sqrt{n}}", language="latex")
            
            st.markdown("##### Final Reported Result:")
            # Use engine's sig fig formatter
            val_str, unc_str = format_sig_figs(mean_val, std_error)
            st.success(f"{val_str} ± {unc_str}")

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
        
    with display_col:
        metrics_col, plot_col = st.columns([1, 2])
        
        x = edited_graph_df["X Values"].dropna().values
        y = edited_graph_df["Y Values"].dropna().values
        
        if len(x) > 1 and len(x) == len(y):
            # Calculate simple regression
            coeffs, cov = np.polyfit(x, y, 1, cov=True)
            slope, intercept = coeffs
            slope_err = np.sqrt(cov[0][0])
            intercept_err = np.sqrt(cov[1][1])
            
            # Calculate R^2
            correlation_matrix = np.corrcoef(x, y)
            r_squared = correlation_matrix[0,1]**2
            
            with metrics_col:
                st.markdown("#### Regression Results")
                
                m_str, m_err_str = format_sig_figs(slope, slope_err)
                c_str, c_err_str = format_sig_figs(intercept, intercept_err)
                
                st.info(f"**Slope (m):** {m_str} ± {m_err_str}")
                st.info(f"**Intercept (c):** {c_str} ± {c_err_str}")
                st.success(f"**R² Value:** {r_squared:.4f}")
                
                st.latex(r"y = mx + c")
                st.latex(f"y = ({m_str})x + ({c_str})")
            
            with plot_col:
                st.markdown("#### Scatter Plot")
                fig, ax = plt.subplots()
                ax.scatter(x, y, label="Raw Data", color="#3182ce")
                ax.plot(x, slope*x + intercept, color="#e53e3e", linestyle="--", label="Best Fit")
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.legend()
                ax.grid(True, linestyle=":", alpha=0.7)
                st.pyplot(fig)

# ==========================================
# TOOL 3: FORMULA BASED
# ==========================================
elif tool == "Formula based":
    st.title("Uncertainty Analysis")
    st.markdown("Enter your formula and measurements below to see the step-by-step derivation.")
    
    input_col, display_col = st.columns([1, 2])
    
    with input_col:
        st.markdown("### 1. Input Parameters")
        formula_input = st.text_input("Formula (e.g., d / t)", "d / t")
        
        # Dynamically extract variables from the formula
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
            col1, col2, col3 = st.columns(3)
            col1.markdown("**Value**")
            col2.markdown("**Uncert (±)**")
            col3.markdown("**Unit**")
            
            for sym in symbols_list:
                # Provide distinct input boxes for each detected variable
                val = col1.number_input(f"{sym} val", value=1.0, key=f"val_{sym}", label_visibility="collapsed")
                unc = col2.number_input(f"{sym} unc", value=0.1, min_value=0.0, format="%.4f", key=f"unc_{sym}", label_visibility="collapsed")
                unit = col3.text_input(f"{sym} unit", value="", key=f"unit_{sym}", label_visibility="collapsed")
                
                variables[sym] = {'val': val, 'uncert': unc, 'unit': unit}
                
        calc_button = st.button("Calculate", type="primary")

    with display_col:
        if valid_formula and symbols_list and calc_button:
            col_res, col_derive = st.columns([1, 1])
            
            # Connect to engine.py
            final_val, final_unc, step_data = propagate_uncertainty(formula_input, variables)
            final_unit = get_final_unit(formula_input, variables)
            val_str, unc_str = format_sig_figs(final_val, final_unc)
            
            with col_res:
                st.markdown("### Final Result")
                st.success(f"**Result:** {val_str} ± {unc_str} {final_unit}")
                
                with st.expander("Show Raw LaTeX (For Reports)"):
                    st.code(f"{val_str} \\pm {unc_str} \\text{{ {final_unit}}}", language="latex")
                    
                st.markdown("### Uncertainty Contributions")
                # Calculate total variance to find percentages
                total_variance = sum([data['contribution'] for data in step_data.values()])
                
                if total_variance > 0:
                    for sym, data in step_data.items():
                        percentage = data['contribution'] / total_variance
                        st.markdown(f"**{sym} contribution: {percentage*100:.1f}%**")
                        st.progress(float(percentage))
                
            with col_derive:
                st.markdown("### Step-by-Step Derivation")
                
                for sym, data in step_data.items():
                    st.markdown(f"Partial derivative with respect to **{sym}**:")
                    # Convert sympy partial derivative to LaTeX
                    latex_deriv = sp.latex(data['symbolic'])
                    st.latex(f"\\frac{{\\partial}}{{\\partial {sym}}} = {latex_deriv}")
                    st.markdown(f"<p style='text-align: center'>Evaluated at {sym} = {data['evaluated']:.4g}</p>", unsafe_allow_html=True)
                    st.markdown("---")
                
                # Optional Monte Carlo Check hook
                with st.expander("Advanced: Run Monte Carlo Verification"):
                    st.info("Running 10,000 simulations based on your inputs...")
                    mc_mean, mc_std, results, median, lower, upper = run_monte_carlo(formula_input, variables)
                    mc_val_str, mc_unc_str = format_sig_figs(mc_mean, mc_std)
                    st.write(f"**Monte Carlo Result:** {mc_val_str} ± {mc_unc_str}")
                    st.write(f"Asymmetric Bounds: +{upper:.4g} / -{lower:.4g}")