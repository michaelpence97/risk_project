import os

def create_scenario_folders(num_scenarios):
    for i in range(1, num_scenarios + 1):
        folder_name = f"Scenario {i:02d}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
        else:
            print(f"Folder {folder_name} already exists")

if __name__ == "__main__":
    create_scenario_folders(27)
