import sympy as sp
import numpy as np
import pint
import math

# Initialize the unit registry
ureg = pint.UnitRegistry()

def get_final_unit(formula_str, variables):
    """Evaluates the formula purely for units and catches dimensional errors."""
    try:
        expr = sp.sympify(formula_str)
        symbols_list = list(expr.free_symbols)
        func = sp.lambdify(symbols_list, expr)
        
        args = [ureg.Quantity(variables[str(sym)]['val'], variables[str(sym)]['unit']) for sym in symbols_list]
        result_qty = func(*args)
        
        return f"{result_qty.units:~P}" 
    except pint.errors.DimensionalityError:
        return "Dimensional Error: Incompatible Units"
    except Exception:
        return ""

def format_sig_figs(value, uncertainty):
    """Rounds uncertainty to 1 sig fig, and matches the value's decimal place."""
    if uncertainty == 0 or not math.isfinite(uncertainty):
        return f"{value:.4g}", f"{uncertainty:.4g}"
        
    order_of_mag = math.floor(math.log10(abs(uncertainty)))
    rounded_uncert = round(uncertainty, -order_of_mag)
    rounded_value = round(value, -order_of_mag)
    
    if order_of_mag < 0:
        decimals = -order_of_mag
        return f"{rounded_value:.{decimals}f}", f"{rounded_uncert:.{decimals}f}"
    else:
        return f"{int(rounded_value)}", f"{int(rounded_uncert)}"

def propagate_uncertainty(formula_str, variables):
    expr = sp.sympify(formula_str)
    symbols = {str(sym): sym for sym in expr.free_symbols}
    
    step_by_step_data = {}
    total_variance = 0
    subs_dict = {symbols[k]: v['val'] for k, v in variables.items()}
    
    for var_name, var_data in variables.items():
        sym = symbols[var_name]
        partial_deriv = sp.diff(expr, sym)
        deriv_val = partial_deriv.evalf(subs=subs_dict)
        contribution = (deriv_val * var_data['uncert'])**2
        total_variance += contribution
        
        step_by_step_data[var_name] = {
            'symbolic': partial_deriv,
            'evaluated': float(deriv_val),
            'contribution': float(contribution)
        }
        
    final_value = float(expr.evalf(subs=subs_dict))
    final_uncertainty = float(sp.sqrt(total_variance))
    
    return final_value, final_uncertainty, step_by_step_data

def run_monte_carlo(formula_str, variables, n_iterations=10000):
    expr = sp.sympify(formula_str)
    symbols_list = list(expr.free_symbols)
    func = sp.lambdify(symbols_list, expr, "numpy")
    
    kwargs = {}
    for sym in symbols_list:
        var_name = str(sym)
        val = variables[var_name]['val']
        unc = variables[var_name]['uncert']
        kwargs[var_name] = np.random.normal(loc=val, scale=unc, size=n_iterations)
        
    args = [kwargs[str(sym)] for sym in symbols_list]
    results = func(*args)
    
    mc_mean = float(np.mean(results))
    mc_std = float(np.std(results))
    
    # Calculate the 16th, 50th (median), and 84th percentiles for asymmetric reporting
    percentiles = np.percentile(results, [16, 50, 84])
    lower_bound = percentiles[1] - percentiles[0]
    upper_bound = percentiles[2] - percentiles[1]
    median = percentiles[1]
    
    # Now correctly returning all 6 expected values!
    return mc_mean, mc_std, results, median, lower_bound, upper_bound