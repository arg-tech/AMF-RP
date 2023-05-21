from flask import request, render_template
from . import application
import json
from app.AMP import relation_identification


@application.route('/somaye', methods=['GET', 'POST'])
def amf_schemes():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        # Preedict Arguemnt Relations from I nodes (propositions) from XAIF file
        response = relation_identification(content)
        # print(response)
        return response
    elif request.method == 'GET':
        return render_template('docs.html')