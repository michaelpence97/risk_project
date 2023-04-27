# HayleyMain.py
script_files = ['Hayley1.py', 'Hayley2.py', 'Hayley3.py']

for script_file in script_files:
    with open(script_file, 'r') as file:
        script_content = file.read()
        exec(script_content)
