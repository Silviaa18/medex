# import the Flask class from the flask module
from flask import Flask, redirect, session, send_file, render_template, Response, request, url_for
from modules.import_scheduler import Scheduler
from url_handlers.models import TableBuilder
import modules.load_data_postgre as ps
from db import connect_db
import pandas as pd
import os
import io

import plotly.express as px

# create the application object
app = Flask(__name__)

app.secret_key = os.urandom(24)
with app.app_context():
    rdb = connect_db()


def check_for_env(key: str, default=None, cast=None):
    if key in os.environ:
        if cast:
            return cast(os.environ.get(key))
        return os.environ.get(key)
    return default


# date and hours to import data
day_of_week = check_for_env('IMPORT_DAY_OF_WEEK', default='mon-sun')
hour = check_for_env('IMPORT_HOUR', default=5)
minute = check_for_env('IMPORT_MINUTE', default=5)


# Import data using function scheduler from package modules
if os.environ.get('IMPORT_DISABLED') is None:
    scheduler = Scheduler(rdb,day_of_week=day_of_week, hour=hour, minute=minute)
    scheduler.start()
    scheduler.stop()


# get all numeric and categorical entities from database
name2, name = ps.get_header(rdb)['Name_ID'][0], ps.get_header(rdb)['measurement'][0]
all_numeric_entities, size_n = ps.get_numeric_entities(rdb)
all_categorical_entities, all_subcategory_entities, size_c, entity = ps.get_categorical_entities(rdb)
all_entities = all_categorical_entities.append(all_numeric_entities, ignore_index=True, sort=False)
all_entities = all_entities.to_dict('index')
all_numeric_entities = all_numeric_entities.to_dict('index')
all_categorical_entities = all_categorical_entities.to_dict('index')
all_measurement = ps.get_measurement(rdb)

if len(all_measurement) < 2:
    block = 'none'
else:
    block = 'block'


database_name = os.environ['POSTGRES_DB']
database='{} data'.format(database_name)
len_numeric = 'number of numerical entities: ' + str(len(all_numeric_entities))
size_numeric = 'the size of the numeric table: ' + str(size_n) + ' rows'
len_categorical = 'number of categorical entities: ' + str(len(all_categorical_entities))
size_categorical = 'the size of the categorical table: ' + str(size_c)+' rows'

numeric_df, error = ps.get_values_scatter_plot('Pod.R231Q_A286V.4h.FDR', 'Pod.R231Q_A286V.12h.FDR', '0', '0', rdb)
numeric_df = numeric_df.rename(columns={"Name_ID": "{}".format(name2), "measurement": "{}".format(name)})
numeric_df['hover_mouse'] = numeric_df[name2] + '<br />' + numeric_df["Gene.Symbol"]
fig = px.scatter(numeric_df, x='Pod.R231Q_A286V.4h.FDR', y='Pod.R231Q_A286V.12h.FDR', hover_name='hover_mouse',
                 template="plotly_white", trendline="ols")
fig.update_layout(
    font=dict(size=16),
    title={
        'text': "Compare values of <b>" + 'Pod.R231Q_A286V.4h.FDR' + "</b> and <b>" + 'Pod.R231Q_A286V.12h.FDR',
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
fig = fig.to_html()

dictOfcolumn = []
table_schema = []
entities = entity['Key'].tolist()
what_table = 'long'
df, error = ps.get_data(entities, what_table, rdb)
df = df.drop(columns=['measurement'])
column = df.columns.tolist()
[dictOfcolumn.append({'data': column[i]}) for i in range(0, len(column))]
[table_schema.append({'data_name': column[i],'column_name': column[i],"default": "","order": 1,"searchable": True}) for i in range(0, len(column))]

# data store for download and so I need work on this and check !!!
class DataStore():

    # table browser
    table_browser_entites = entity['Key'].tolist()
    table_browser_what_table = 'long'
    csv = df.to_csv(index=False)
    dict = df.to_dict("records")
    table_schema = table_schema
    table_browser_column = column
    table_browser_column2 = dictOfcolumn


    # Basic Stats
    basic_stats_numeric_entities = None
    basic_stats_measurement_n = None
    basic_stats_instance_n = None
    basic_stats_numeric_results_n = None

    basic_stats_categorical_entities = None
    basic_stats_measurement_c = None
    basic_stats_instance_c = None
    basic_stats_numeric_results_c = None


    # Scatter plot
    scatter_plot_x_axis = "Pod.R231Q_a286V.4h.FDR"
    scatter_plot_y_axis = "Pod.R231Q_a286V.1h.FDR"
    scatter_plot_x_measurement = '0'
    scatter_plot_y_measurement = '0'
    scatter_plot_categorical_entities = ""
    scatter_plot_subcategory_entities = []
    scatter_plot_how_to_plot = 'linear'
    scatter_plot_log_x = None
    scatter_plot_log_y = None
    scatter_plot_add_group_by = False
    scatter_plot_fig = fig

    # Barchart
    barchart_measurement = None
    barchart_all_measurement = None
    barchart_categorical_entities = None
    barchart_subcategory_entities = None
    barchart_fig = None

    # Histogram
    histogram_number_of_bins = None
    histogram_numeric_entities = None
    histogram_categorical_entities = None
    histogram_subcategory_entities = None
    histogram_measurement = None
    histogram_fig = None

    # Boxplot

    # Heatmap
    heatmap_numeric_entities = None
    heatmap_measurement = None
    heatmap_plot_series = None

    # Clustering
    clustering_entities = None
    clustering_cluster_info = None
    clustering_all_present = None
    clustering_any_present = None
    clustering_fig = None

    # Coplots
    coplots_how_to_plot = None
    coplots_select_scale = None
    coplots_category11 = None
    coplots_category22 = None
    coplots_category1 = None
    coplots_category2 = None
    coplots_x_axis = None
    coplots_y_axis = None
    coplots_x_measurement = None
    coplots_y_measurement = None
    coplots_fig = None

table_builder = TableBuilder()
data = DataStore()

# Urls in the 'url_handlers' directory (one file for each new url)
# import a Blueprint
from url_handlers.data import data_page
from url_handlers.basic_stats import basic_stats_page
from url_handlers.histogram import histogram_page
from url_handlers.boxplot import boxplot_page
from url_handlers.scatter_plot import scatter_plot_page
from url_handlers.barchart import barchart_page
from url_handlers.heatmap import heatmap_plot_page
from url_handlers.clustering_pl import clustering_plot_page
from url_handlers.coplots_pl import coplots_plot_page
from url_handlers.logout import logout_page

# register blueprints here:
app.register_blueprint(data_page)
app.register_blueprint(logout_page)
app.register_blueprint(basic_stats_page)
app.register_blueprint(histogram_page)
app.register_blueprint(boxplot_page)
app.register_blueprint(scatter_plot_page)
app.register_blueprint(barchart_page)
app.register_blueprint(heatmap_plot_page)
app.register_blueprint(clustering_plot_page)
app.register_blueprint(coplots_plot_page)


# Direct to Data browser website during opening the program.
@app.route('/', methods=['GET'])
def login():
    return redirect('/data')

@app.route("/download", methods=['GET', 'POST'])
def download():

    csv=data.csv

    # Create a string buffer
    buf_str = io.StringIO(csv)

    # Create a bytes buffer from the string buffer
    buf_byt = io.BytesIO(buf_str.read().encode("utf-8"))

    # Return the CSV data as an attachment
    return send_file(buf_byt,
                     mimetype="text/csv",
                     as_attachment=True,
                     attachment_filename="data.csv")


def main():

    return app
