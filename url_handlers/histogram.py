from flask import Blueprint, render_template, request
import pandas as pd

import data_warehouse.redis_rwh as rwh

histogram_page = Blueprint('histogram', __name__,
                           template_folder='templates')


@histogram_page.route('/histogram', methods=['GET'])
def get_statistics():
    # this import has to be here!!
    from webserver import get_db
    rdb = get_db()
    all_numeric_entities = rwh.get_numeric_entities(rdb)
    all_categorical_entities = rwh.get_categorical_entities(rdb)

    return render_template('histogram.html', categorical_entities=all_categorical_entities,
                           numeric_entities=all_numeric_entities)


@histogram_page.route('/histogram', methods=['POST'])
def post_statistics():
    # this import has to be here!!
    from webserver import get_db
    rdb = get_db()
    all_numeric_entities = rwh.get_numeric_entities(rdb)
    all_categorical_entities = rwh.get_categorical_entities(rdb)

    entity = request.form.get('entity')
    group_by = request.form.get('group_by')
    #number_of_bins = request.form.get('number_of_bins')
    if not entity or not group_by or entity == "Choose entity" or group_by == "Choose entity":
        error = "Please select entity and group_by"

        return render_template('histogram.html', categorical_entities=all_categorical_entities,
                               numeric_entities=all_numeric_entities,
                               error=error, )

    numeric_df = rwh.get_joined_numeric_values([entity], rdb)
    categorical_df = rwh.get_joined_categorical_values([group_by], rdb)
    merged_df = pd.merge(numeric_df, categorical_df, how='inner', on='patient_id')
    min_val = numeric_df[entity].min()
    max_val = numeric_df[entity].max()

    groups = set(categorical_df[group_by].values.tolist())
    plot_series = []
    for group in groups:
        df = merged_df.loc[merged_df[group_by] == group]
        values = df[entity].values.tolist()
        if (values):
            plot_series.append({
                'x'       : values,
                'type'    : "histogram",
                'opacity' : 0.5,
                'name'    : group,
                # 'autobinx': 'false',
                # 'xbins': number_of_bins,
                # 'xbins'   : {
                    # 'end': 4,
                    #'size': number_of_bins,
                    # 'start': -3.2
                    #}
                })

    return render_template('histogram.html',
                           categorical_entities=all_categorical_entities,
                           numeric_entities=all_numeric_entities,
                           selected_entity=entity,
                           group_by=group_by,
                           plot_series=plot_series,
                           min_val=min_val,
                           max_val=max_val
                           #number_of_bins=number_of_bins,
                           )
