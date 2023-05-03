import {switch_nav_item} from "../utility/nav.mjs";
import {get_radio_input_by_name} from "../utility/dom.mjs";
import {get_selected_items, get_selected_measurements, show_collapsed} from "../utility/misc.mjs";
import 'datatables.net-dt/css/jquery.dataTables.min.css';

import {configure_entity_selection} from "../utility/entity_selection.mjs";
import {configure_multiple_measurement_select} from "../services/measurement.mjs";

import Datatable from 'datatables.net-dt';
import {ALL_ENTITIES} from "../services/entity.mjs";
import '../../examples/diabetes_prediction_dataset.csv'


async function init() {
    switch_nav_item('risk_score');


}

function getResults(): void {
    const csvFilePath = 'diabetes_prediction_dataset.csv';

    // Fetch the CSV file
    fetch(csvFilePath)
        .then(response => response.text())
        .then(data => {
            // Split the data by rows
            const rows = data.split('\n');

            // Create a table element
            const table = document.createElement('table');
            table.classList.add('table');

            // Create a table header row
            const headerRow = document.createElement('tr');

            // Split the first row (header row) by commas to get column names
            const headers = rows[0].split(',');

            // Define the column names
            const columnNames = [
                'gender',
                'age',
                'hypertension',
                'heart_disease',
                'smoking_history',
                'bmi',
                'HbA1c_level',
                'blood_glucose_level',
                'diabetes'
            ];

            // Create table header cells based on column names
            columnNames.forEach(columnName => {
                const th = document.createElement('th');
                th.textContent = columnName;
                headerRow.appendChild(th);
            });

            // Add the header row to the table
            table.appendChild(headerRow);

            // Iterate through the remaining rows (data rows)
            for (let i = 1; i < rows.length; i++) {
                const rowData = rows[i].split(',');

                // Create a table row for each data row
                const row = document.createElement('tr');

                // Create table cells and add them to the row
                columnNames.forEach((columnName, index) => {
                    const td = document.createElement('td');
                    td.textContent = rowData[index];
                    row.appendChild(td);
                });

                // Add the data row to the table
                table.appendChild(row);
            }

            // Append the table to a container element in your HTML
            const container = document.getElementById('table-container');
            container.innerHTML = '';
            container.appendChild(table);
        })
        .catch(error => {
            console.error('Error fetching CSV file:', error);
        });
    document.addEventListener("DOMContentLoaded", init);
}