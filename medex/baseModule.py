from flask import Flask, jsonify
from medex.database_info import DatabaseInfoService, db_session
from medex.dto.database_info import DatabaseInfo

app = Flask(__name__)
database_info_service = DatabaseInfoService(db_session)

@app.route('/patients')
def get_patients():
    with db.session() as session:
        patients = session.query(Patient).all()
    return jsonify(patients)

@app.route('/entities/<entity_type>')
def get_entities(entity_type):
    entity_type = EntityType[entity_type.upper()]
    with db.session() as session:
        entities = session.query(TableNumerical).filter_by(name_type_key=entity_type.value).all()
    return jsonify(entities)


@app.route('/api/database_info', methods=['GET'])
def get_database_info():
    database_info = database_info_service.get()
    return jsonify({
        'number_of_patients': database_info.number_of_patients,
        'number_of_numerical_entities': database_info.number_of_numerical_entities,
        'number_of_categorical_entities': database_info.number_of_categorical_entities,
        'number_of_date_entities': database_info.number_of_date_entities,
        'number_of_numerical_data_items': database_info.number_of_numerical_data_items,
        'number_of_categorical_data_items': database_info.number_of_categorical_data_items,
        'number_of_date_data_items': database_info.number_of_date_data_items
    })

if __name__ == '__main__':
    app.run()
