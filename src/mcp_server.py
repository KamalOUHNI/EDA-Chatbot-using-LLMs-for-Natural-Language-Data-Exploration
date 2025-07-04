from mcp.server.fastmcp import FastMCP
import pandas as pd
import urllib.parse
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from typing import Optional, List
import numpy as np
import seaborn as sns

# Initialize MCP server
mcp = FastMCP("mcp_server")

db = None
engine = None
current_df = None
current_table = None
connection_uri = None

@mcp.tool()
def connect_to_sql_server(server: str = "localhost") -> str:
    """
    Connect to SQL Server database.
    
    Args:
        server: SQL Server instance (default: localhost)
        database: Database name (default: LLMbench)
        driver: ODBC driver name (default: ODBC Driver 17 for SQL Server)
    
    Returns:
        Success message with available tables or error
    """
    global db, engine, connection_uri
    
    try:
        params = urllib.parse.quote_plus(
            f"DRIVER={{{'ODBC Driver 17 for SQL Server'}}};"
            f"SERVER={server};" 
            f"DATABASE={'LLMbench'};" 
            "trusted_connection=yes;"
        )
        connection_uri = f"mssql+pyodbc:///?odbc_connect={params}"
        
        # SQLDatabase and engine
        db = SQLDatabase.from_uri(connection_uri)
        engine = create_engine(connection_uri)
        
        #  available tables
        tables = db.get_usable_table_names()
        
        return f"""Successfully connected to SQL Server:
- Server: {server}
- Database: {'LLMbench'}
- Dialect: {db.dialect}
- Available tables: {', '.join(tables)}

Use select_table to choose a table for analysis."""
        
    except Exception as e:
        return f"Error connecting to SQL Server: {str(e)}"

@mcp.tool()
def get_available_tables(Database : str) -> str:
    """
    Get list of all usable tables in the connected database.
    Args:
        DataBase: Name of the Database
    Returns:
        List of table names as string
    """
    global db
    if db is None:
        return "Error: No database connection. Use connect_to_sql_server first."
    
    try:
        tables = db.get_usable_table_names()
        if not tables:
            return "No tables found in the database."
        
        return f"Available tables: {', '.join(tables)}"
    except Exception as e:
        return f"Error retrieving tables: {str(e)}"

@mcp.tool()
def select_table(table_name: str) -> str:
    """
    Select a table from the database and load it as DataFrame.
    
    Args:
        table_name: Name of the table to select
    
    Returns:
        Success message with table info or error
    """
    global db, engine, current_df, current_table
    if db is None or engine is None:
        return "Error: No database connection. Use connect_to_sql_server first."
    
    try:
        available_tables = db.get_usable_table_names()
        if table_name not in available_tables:
            return f"Error: Table '{table_name}' is not available."
        
        # Load table into DataFrame
        current_df = pd.read_sql_table(table_name, con=engine)
        current_table = table_name
        
        info = f"""Successfully loaded table '{table_name}':
- Shape: {current_df.shape}
- Columns: {list(current_df.columns)}
- Data types: {dict(current_df.dtypes)}
- Memory usage: {current_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Use get_table_preview to see sample data."""
        return info
    except Exception as e:
        return f"Error loading table '{table_name}': {str(e)}"

@mcp.tool()
def get_table_preview(num_rows: int = 5) -> str:
    """
    Get a preview of the currently selected table.
    
    Args:
        num_rows: Number of rows to preview (default: 5)
    
    Returns:
        Table preview as string
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        preview = current_df.head(num_rows).to_string()
        return f"Preview of table '{current_table}' (first {num_rows} rows):\n{preview}"
    except Exception as e:
        return f"Error getting table preview: {str(e)}"

@mcp.tool()
def get_table_info(dummy : str) -> str:
    """
    Get detailed information about the currently selected table.
    
    Returns:
        Table information as string
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        #  info
        info_lines = [
            f"Table: {current_table}",
            f"Shape: {current_df.shape} (rows, columns)",
            f"Columns: {list(current_df.columns)}",
            "",
            "Data Types:",
            current_df.dtypes.to_string(),
            "",
            "Missing Values:",
            current_df.isnull().sum().to_string(),
            "",
            f"Numeric Columns: {current_df.select_dtypes(include=['number']).columns.tolist()}",
            f"Text Columns: {current_df.select_dtypes(include=['object']).columns.tolist()}",
            f"DateTime Columns: {current_df.select_dtypes(include=['datetime']).columns.tolist()}"
        ]
        
        return "\n".join(info_lines)
    except Exception as e:
        return f"Error getting table info: {str(e)}"
@mcp.tool() 
def get_missing_value_summary(dummy: str) -> str:
    """
    Get a quick summary of missing values in the currently selected table.
    
    Returns:
        Brief missing values summary as string
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        total_missing = current_df.isnull().sum().sum()
        total_cells = current_df.shape[0] * current_df.shape[1]
        
        if total_missing == 0:
            return f" {current_table}: No missing values found"
        
        missing_percentage = (total_missing / total_cells) * 100
        cols_with_missing = (current_df.isnull().sum() > 0).sum()
        
        return f"{current_table}: {total_missing:,} missing values ({missing_percentage:.1f}%) across {cols_with_missing} columns"
        
    except Exception as e:
        return f"Error getting missing values summary: {str(e)}"

# UNIVARIATE NON-GRAPHICAL TOOLS

@mcp.tool()
def univariate_numeric_summary(column: str) -> str:
    """
    Get comprehensive descriptive statistics for a numeric column.
    
    Args:
        column: Name of the numeric column to analyze
    
    Returns:
        Detailed statistical summary as string
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        if column not in current_df.columns:
            return f"Error: Column '{column}' not found in table."
        
        if not pd.api.types.is_numeric_dtype(current_df[column]):
            return f"Error: Column '{column}' is not numeric."
        
        col_data = current_df[column].dropna()
        
        if len(col_data) == 0:
            return f"Error: Column '{column}' has no non-null values."
        
        # statistics
        stats = col_data.describe()
        
        from scipy import stats as scipy_stats
        
        info_lines = [
            f"Univariate Analysis: {column}",
            "=" * 40,
            "",
            "Basic Statistics:",
            "-" * 20,
            f"Count: {len(col_data):,}",
            f"Missing: {current_df[column].isnull().sum():,}",
            f"Mean: {stats['mean']:.4f}",
            f"Median: {stats['50%']:.4f}",
            f"Mode: {col_data.mode().iloc[0] if not col_data.mode().empty else 'No mode'}",
            f"Standard Deviation: {stats['std']:.4f}",
            f"Variance: {col_data.var():.4f}",
            f"Range: {stats['max'] - stats['min']:.4f}",
            f"Min: {stats['min']:.4f}",
            f"Max: {stats['max']:.4f}",
            "",
            "Quartiles:",
            "-" * 10,
            f"Q1 (25%): {stats['25%']:.4f}",
            f"Q2 (50%): {stats['50%']:.4f}",
            f"Q3 (75%): {stats['75%']:.4f}",
            f"IQR: {stats['75%'] - stats['25%']:.4f}",
            "",
            "Shape Statistics:",
            "-" * 15,
            f"Skewness: {scipy_stats.skew(col_data):.4f}",
            f"Kurtosis: {scipy_stats.kurtosis(col_data):.4f}",
            "",
            "Outlier Detection (IQR method):",
            "-" * 30
        ]
        
        # Outlier detection
        Q1, Q3 = stats['25%'], stats['75%']
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        
        info_lines.extend([
            f"Lower fence: {lower_bound:.4f}",
            f"Upper fence: {upper_bound:.4f}",
            f"Outliers: {len(outliers)} ({len(outliers)/len(col_data)*100:.1f}%)"
        ])
        
        if len(outliers) > 0 and len(outliers) <= 10:
            info_lines.append(f"Outlier values: {sorted(outliers.tolist())}")
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error analyzing numeric column: {str(e)}"


@mcp.tool()
def univariate_categorical_summary(column: str) -> str:
    """
    Get comprehensive summary for a categorical column.
    
    Args:
        column: Name of the categorical column to analyze
    
    Returns:
        Detailed categorical summary as string
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        if column not in current_df.columns:
            return f"Error: Column '{column}' not found in table."
        
        col_data = current_df[column]
        non_null_data = col_data.dropna()
        
        value_counts = non_null_data.value_counts()
        value_props = non_null_data.value_counts(normalize=True)
        
        info_lines = [
            f"Categorical Analysis: {column}",
            "=" * 40,
            "",
            "Basic Statistics:",
            "-" * 20,
            f"Total Count: {len(col_data):,}",
            f"Non-null Count: {len(non_null_data):,}",
            f"Missing Count: {col_data.isnull().sum():,}",
            f"Unique Categories: {len(value_counts)}",
            f"Most Frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]:,} times)",
            f"Least Frequent: '{value_counts.index[-1]}' ({value_counts.iloc[-1]:,} times)",
            "",
            "Category Distribution:",
            "-" * 25
        ]
        
        top_n = min(15, len(value_counts))
        for i in range(top_n):
            category = value_counts.index[i]
            count = value_counts.iloc[i]
            prop = value_props.iloc[i]
            info_lines.append(f"{category}: {count:,} ({prop:.1%})")
        
        if len(value_counts) > top_n:
            remaining = len(value_counts) - top_n
            info_lines.append(f"... and {remaining} more categories")
        
        info_lines.extend([
            "",
            "Concentration Metrics:",
            "-" * 20,
            f"Entropy: {-sum(p * np.log2(p) for p in value_props if p > 0):.4f}",
            f"Gini Index: {1 - sum(p**2 for p in value_props):.4f}",
            f"Top 1 concentration: {value_props.iloc[0]:.1%}",
            f"Top 3 concentration: {value_props.head(3).sum():.1%}" if len(value_props) >= 3 else "",
            f"Top 5 concentration: {value_props.head(5).sum():.1%}" if len(value_props) >= 5 else ""
        ])
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error analyzing categorical column: {str(e)}"


# UNIVARIATE GRAPHICAL TOOLS

@mcp.tool()
def create_histogram(column: str) -> str:
    """
    Create histogram visualization for a numeric column (uses 30 bins by default).
    
    Args:
        column: Name of the numeric column
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if column not in current_df.columns:
            return f"Error: Column '{column}' not found in table."
        
        if not pd.api.types.is_numeric_dtype(current_df[column]):
            return f"Error: Column '{column}' is not numeric."
        
        plt.figure(figsize=(10, 6))
        
        plt.hist(current_df[column].dropna(), bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        from scipy import stats
        data = current_df[column].dropna()
        x = np.linspace(data.min(), data.max(), 100)
        kde = stats.gaussian_kde(data)
        plt.plot(x, kde(x), 'r-', linewidth=2, label='Density')
        
        plt.title(f'Histogram: {column}')
        plt.xlabel(column)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filename = f"{current_table}_{column}_histogram.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Histogram saved as {filename}"
        
    except Exception as e:
        return f"Error creating histogram: {str(e)}"


@mcp.tool()
def create_boxplot(column: str) -> str:
    """
    Create boxplot visualization for a numeric column.
    
    Args:
        column: Name of the numeric column
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if column not in current_df.columns:
            return f"Error: Column '{column}' not found in table."
        
        if not pd.api.types.is_numeric_dtype(current_df[column]):
            return f"Error: Column '{column}' is not numeric."
        
        plt.figure(figsize=(10, 6))
        
        box = plt.boxplot(current_df[column].dropna(), vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('lightblue')
        
        plt.title(f'Boxplot: {column}')
        plt.ylabel(column)
        plt.grid(True, alpha=0.3)
        
        filename = f"{current_table}_{column}_boxplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Boxplot saved as {filename}"
        
    except Exception as e:
        return f"Error creating boxplot: {str(e)}"


@mcp.tool()
def create_bar_chart(column: str) -> str:
    """
    Create bar chart for categorical column (shows top 20 categories).
    
    Args:
        column: Name of the categorical column
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        
        if column not in current_df.columns:
            return f"Error: Column '{column}' not found in table."
        
        value_counts = current_df[column].value_counts().head(20)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral')
        
        plt.title(f'Bar Chart: {column} (Top {len(value_counts)})')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
        
        for bar, value in zip(bars, value_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(value_counts)/50, 
                    f'{value:,}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        filename = f"{current_table}_{column}_barchart.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Bar chart saved as {filename}"
        
    except Exception as e:
        return f"Error creating bar chart: {str(e)}"



@mcp.tool()
def correlation_analysis(method: str) -> str:
    """
    Calculate correlation matrix for all numeric columns.
    
    Args:
        method: Correlation method ("pearson", "spearman", or "kendall")
    
    Returns:
        Correlation analysis results
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        if method not in ["pearson", "spearman", "kendall"]:
            return "Error: Method must be 'pearson', 'spearman', or 'kendall'"
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Error: Need at least 2 numeric columns for correlation analysis."
        
        corr_matrix = current_df[numeric_cols].corr(method=method)
        
        info_lines = [
            f"Correlation Analysis ({method.title()}):",
            "=" * 50,
            "",
            "Correlation Matrix:",
            "-" * 20,
            corr_matrix.round(4).to_string(),
            "",
            "Strong Correlations (|r| > 0.7):",
            "-" * 35
        ]
        
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strong_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        if strong_corrs:
            for col1, col2, corr_val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                info_lines.append(f"{col1} ↔ {col2}: {corr_val:.4f}")
        else:
            info_lines.append("No strong correlations found.")
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error in correlation analysis: {str(e)}"


@mcp.tool()
def crosstab_analysis(columns: str) -> str:
    """
    Create cross-tabulation analysis between two categorical columns.
    
    Args:
        columns: Two column names separated by comma (e.g., "col1,col2")
    
    Returns:
        Cross-tabulation results with chi-square test
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        cols = [col.strip() for col in columns.split(",")]
        if len(cols) != 2:
            return "Error: Please provide exactly two column names separated by comma."
        
        col1, col2 = cols
        
        if col1 not in current_df.columns or col2 not in current_df.columns:
            return f"Error: One or both columns not found in table."
        
        crosstab = pd.crosstab(current_df[col1], current_df[col2], margins=True)
        
        # Chi-square test
        from scipy.stats import chi2_contingency
        
        chi2_table = crosstab.iloc[:-1, :-1]
        chi2, p_value, dof, expected = chi2_contingency(chi2_table)
        
        info_lines = [
            f"Cross-tabulation: {col1} vs {col2}",
            "=" * 50,
            "",
            "Cross-tabulation Table:",
            "-" * 25,
            crosstab.to_string(),
            "",
            "Statistical Test Results:",
            "-" * 25,
            f"Chi-square statistic: {chi2:.4f}",
            f"p-value: {p_value:.6f}",
            f"Degrees of freedom: {dof}",
            f"Significant association: {'Yes' if p_value < 0.05 else 'No'} (α = 0.05)"
        ]
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error in cross-tabulation analysis: {str(e)}"


# MULTIVARIATE GRAPHICAL TOOLS

@mcp.tool()
def create_correlation_heatmap(method: str) -> str:
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        method: Correlation method ("pearson", "spearman", or "kendall")
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if method not in ["pearson", "spearman", "kendall"]:
            return "Error: Method must be 'pearson', 'spearman', or 'kendall'"
        
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return "Error: Need at least 2 numeric columns for correlation heatmap."
        
        corr_matrix = current_df[numeric_cols].corr(method=method)
        
        plt.figure(figsize=(12, 10))
        
        #heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        
        plt.title(f'Correlation Heatmap ({method.title()})')
        plt.tight_layout()
        
        filename = f"{current_table}_correlation_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f" Correlation heatmap saved as {filename}"
        
    except Exception as e:
        return f"Error creating correlation heatmap: {str(e)}"


@mcp.tool()
def create_scatter_plot(columns: str) -> str:
    """
    Create scatter plot between two numeric columns.
    
    Args:
        columns: Two or three column names separated by comma (e.g., "x_col,y_col" or "x_col,y_col,color_col")
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        
        cols = [col.strip() for col in columns.split(",")]
        if len(cols) not in [2, 3]:
            return "Error: Please provide 2 or 3 column names separated by comma."
        
        x_col, y_col = cols[0], cols[1]
        color_col = cols[2] if len(cols) == 3 else None
        
        if x_col not in current_df.columns or y_col not in current_df.columns:
            return f"Error: Column(s) not found in table."
        
        if not pd.api.types.is_numeric_dtype(current_df[x_col]) or not pd.api.types.is_numeric_dtype(current_df[y_col]):
            return f"Error: Both x and y columns must be numeric."
        
        plt.figure(figsize=(10, 8))
        
        if color_col and color_col in current_df.columns:
            # Colored scatter plot
            if pd.api.types.is_numeric_dtype(current_df[color_col]):
                scatter = plt.scatter(current_df[x_col], current_df[y_col], 
                                    c=current_df[color_col], cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label=color_col)
            else:
                # Categorical color column
                categories = current_df[color_col].unique()[:10]  # Limit to 10 categories
                colors = plt.cm.Set1(np.linspace(0, 1, len(categories)))
                for i, cat in enumerate(categories):
                    mask = current_df[color_col] == cat
                    plt.scatter(current_df[mask][x_col], current_df[mask][y_col], 
                              color=colors[i], label=str(cat), alpha=0.6)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(current_df[x_col], current_df[y_col], alpha=0.6, color='blue')
        
        # Add trend line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            current_df[x_col].dropna(), current_df[y_col].dropna())
        line = slope * current_df[x_col] + intercept
        plt.plot(current_df[x_col], line, 'r--', alpha=0.8, 
                label=f'Trend line (r={r_value:.3f})')
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        filename = f"{current_table}_{x_col}_vs_{y_col}_scatter.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f" Scatter plot saved as {filename}"
        
    except Exception as e:
        return f"Error creating scatter plot: {str(e)}"


@mcp.tool()
def create_pairplot(columns: str) -> str:
    """
    Create pairwise scatter plots for numeric columns.
    
    Args:
        columns: Comma-separated column names or "all" for all numeric columns (max 1000 rows sampled)
    
    Returns:
        Status message and saves plot
    """
    global current_df, current_table
    if current_df is None:
        return "Error: No table selected. Use select_table first."
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if columns == "all":
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns
        else:
            numeric_cols = [col.strip() for col in columns.split(",")]
            numeric_cols = [col for col in numeric_cols if col in current_df.columns]
        
        if len(numeric_cols) < 2:
            return "Error: Need at least 2 numeric columns for pairplot."
        
        if len(numeric_cols) > 6:
            return "Error: Too many columns (max 6) for pairplot. Specify fewer columns."
        
        plot_data = current_df[numeric_cols]
        sample_size = 1000
        if len(plot_data) > sample_size:
            plot_data = plot_data.sample(n=sample_size, random_state=42)
        
        # Create pairplot
        g = sns.pairplot(plot_data, diag_kind='hist', plot_kws={'alpha': 0.6})
        g.fig.suptitle(f'Pairplot: {", ".join(numeric_cols)}', y=1.02)
        
        filename = f"{current_table}_pairplot.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return f"Pairplot saved as {filename} (using {len(plot_data):,} rows)"
        
    except Exception as e:
        return f"Error creating pairplot: {str(e)}"
if __name__ == "__main__":
    print("Starting MCP Server ...")
    mcp.run(transport="stdio")