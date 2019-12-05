from flask import Blueprint, render_template, request
import pandas as pd

import data_warehouse.redis_rwh as rwh

boxplot_page = Blueprint('boxplot', __name__,
                         template_folder='templates')


@boxplot_page.route('/boxplot', methods=['GET'])
def get_boxplots():
    # this import has to be here!!
    from webserver import get_db
    rdb = get_db()
    all_numeric_entities = rwh.get_numeric_entities(rdb)
    all_categorical_entities = rwh.get_categorical_entities(rdb)

    return render_template('boxplot.html', categorical_entities=all_categorical_entities,
                           numeric_entities=all_numeric_entities)


@boxplot_page.route('/boxplot', methods=['POST'])
def post_boxplots():
    # this import has to be here!!
    from webserver import get_db
    rdb = get_db()
    all_numeric_entities = rwh.get_numeric_entities(rdb)
    all_categorical_entities = rwh.get_categorical_entities(rdb)
    entity = request.form.get('entity')
    group_by = request.form.get('group_by')

    numeric_df = rwh.get_joined_numeric_values([entity], rdb)
    categorical_df = rwh.get_joined_categorical_values([group_by], rdb)
    merged_df = pd.merge(numeric_df, categorical_df, how='inner', on='patient_id')
    min_val = numeric_df[entity].min()
    max_val = numeric_df[entity].max()

    groups = set(categorical_df[group_by].values.tolist())
    plot_series = []
    for group in sorted(groups):
        df = merged_df.loc[merged_df[group_by] == group]
        # print(df)
        values = df[entity].values.tolist()
        # print(entity, group, values[:10])
        if (values):
            plot_series.append({
                'y'   : values,
                'type': "box",
                # 'opacity': 0.5,
                'name': group,
                })

    return render_template('boxplot.html',
                           categorical_entities=all_categorical_entities,
                           numeric_entities=all_numeric_entities,
                           selected_entity=entity,
                           group_by=group_by,
                           plot_series=plot_series,
                           min_val=min_val,
                           max_val=max_val,
                           )
