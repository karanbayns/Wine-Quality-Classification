import pandas as pd
import click
import altair as alt

@click.command()
@click.argument('path_read', type = str)
@click.argument('path_save', type = str)
def main(path_read, path_save):  
    train_df = pd.read_csv(path_read)
    feature_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                       'pH', 'sulphates', 'alcohol']  
    train_df[feature_columns].describe().round(3).to_csv(path_save+"/summary_table.csv")
    
    # Split into X and y
    X_train = train_df[feature_columns]
    y_train = train_df['quality_binary']
    
    # Create correlation data frame in long format
    corr_df = pd.concat([X_train, y_train], axis=1).corr('spearman').stack().reset_index()
    corr_df.columns = ['feature_1', 'feature_2', 'correlation']
    corr_df.loc[corr_df['correlation'] == 1, 'correlation'] = 0 # Remove diagonal

    # Create correlation heatmap
    corr_heatmap = alt.Chart(
    corr_df,
    title = 'Wine Quality Correlation Heatmap').mark_rect().encode(
    x = alt.X('feature_1').title('Feature 1'),
    y = alt.Y('feature_2').title('Feature 2'),
    color = alt.Color('correlation').scale(scheme = 'blueorange', domain = (-1,1)).title('Correlation'),
    tooltip = alt.Tooltip('correlation:Q', format = '.2f'))

    # Save correlation heatmap
    corr_heatmap.save(path_save+"/eda_heatmap.png")

    # Isolate target and correlates
    dist_feats = ['quality_binary', 'alcohol', 'sulphates', 'volatile acidity']

    # Create density data frame
    dist_df = pd.concat([X_train, y_train], axis=1)
    dist_df = dist_df[dist_feats]

    # Replace boolean with descriptive strings
    dist_df['quality_binary'] = dist_df['quality_binary'].map({
                                True: 'High Quality Wine',
                                False: 'Low Quality Wine'})

    # Create overlaid histograms for each correlated feature
    feature_hists = alt.Chart(dist_df).mark_bar(opacity = 0.5).encode(
                    x = alt.X(alt.repeat('column')).type('quantitative').bin(maxbins = 25).axis(format = '.1f'),
                    y = alt.Y('count()').stack(False),
                    color = alt.Color('quality_binary:N').title('Wine Quality')
                    ).properties(
                    width = 250,
                    height = 200,
                    ).repeat(
                    column = ['alcohol', 'sulphates', 'volatile acidity']
                    ).resolve_scale(
                    y = 'shared')

    # Display histograms
    feature_hists.save(path_save+"/eda_hists.png")

if __name__ == '__main__':
    main()