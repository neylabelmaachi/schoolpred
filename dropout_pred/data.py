import os
import pandas as pd

# Column from baseline to use
BASELINE_COLS = {
    "hhid": "hhid",
    # Parents
    "gendercode":"Gender",
    "a7_1": "Mother still living",
    "a10_1": "Father still living",
    "a13_1": "Age, in years",
    "a18_1": "Marital Status",
    # Parents Education
    "a14_1_1": "darija",
    "a14_1_2": "class_arabic",
    "a14_1_3": "french",
    "a14_1_4": "amazygh",
    "a15_1_1": "read_one_lang",
    "a15_1_2": "write_one_lang",
    "a15_1_3": "no_read_write",
    "a17_1_cycle": "parents_level_ed",
    "a16_1": "work_activity",
    # Housing
    "im1": "number_of_person_in_hh",
    "b3_1": "type_of_housing",
    "b10_29": "automobiles",
    "b10_20": "mobile_phones",
    "b10_4": "satellite_receivers",
    "b7_11": "no_water",
    "b7_1": "individual_water_net",
    "b6_1": "electrical_net_co",
    # Children status
    "schoolid": "school_id",
    "d5_1": "child_enrollment",
    "d12_1_cycle": "class_when_dropout",
    # Geography
    "schoolid": "school_id",
    "id8":"region",
    "id7":"province",
    # Target(y)
    "d8_1":"age_dropout",
}

# Columns form test results to use
TEST_RESULTS = {
    # Test
    "t1": "done_test",
    "t3_1": "digit_recognition_res",
    "t4_1": "number_recognition_res",
    "t5_1": "subtraction_res",
    "t6": "division_res",
}

class DropoutPred:
    def __init__():
        pass
    
    @classmethod
    def get_data(cls):
        """
        Returns a python dict
        keys are the filenames without suffixes and prefixs
        values are pandas DataFrames loaded from CSV files
        """
        abs_path = os.path.dirname(__file__)
        csv_path = os.path.join(abs_path, "..", "raw_data")
    
        file_names = [file_name for file_name in os.listdir(csv_path) if not file_name.startswith(".")]        
        
        # Create list of dict keys
        key_names = []
        for file_name in file_names:
            # Skip configuration files
            if file_name.startswith("."):
                continue
            
            # Skip non-csv files
            if not file_name.endswith(".csv"):
                continue
            
            # Remove prefixes
            if file_name.startswith("Morocco_CCT_Education_"):
                file_name = file_name.replace("Morocco_CCT_Education_", "")
                
            # Remove suffixes
            if file_name.endswith(".csv"):
                file_name = file_name.replace(".csv", "")

            key_names.append(file_name)

        data = {}
        for key, file_name in zip(sorted(key_names), sorted(file_names)):
            # print(f"key: {key.lower()}")
            # print(f"file_name: {file_name}")
            data[key.lower()] = pd.read_csv(os.path.join(csv_path, file_name), low_memory=False)

        return data
    
    @classmethod
    def get_training_data(cls, done_test=False):
        """
        Returns a DataFrame with columns from
        Math test results and Baseline household data
        and renames encoded columns
        """
        data = cls.get_data()
        baseline_household = data["baseline_household"].copy()
        test_results = data["child_math_test_results"].copy()   
        
        # Filter dataframe
        col_to_drop = ["year","round","endline", "studycode","unitobs","countrycode","ruralcode", "gendercode", "agecode", 't3_2', 't4_2', 't5_rep1', 't5_rep2', 't5_2', 't6_rep', 'hhid_endline', "surveycode", "region", "province", "t2_1", "t2_2", "prenom_enf_test", "schoolid", "benef", "id_enf_test"]
        test_results.drop(columns=col_to_drop, inplace=True)
        baseline_household = baseline_household[BASELINE_COLS.keys()]

        # Rename columns
        test_results.rename(columns=TEST_RESULTS,inplace=True)
        baseline_household.rename(columns=BASELINE_COLS, inplace=True)
        
        # Merge 
        baseline_with_test_results = pd.merge(baseline_household, test_results, on="hhid")
        
        if done_test:
            baseline_with_test_results = baseline_with_test_results[baseline_with_test_results["done_test"].notnull()]

        return baseline_with_test_results

    @classmethod
    def ping(cls):
        """
        You call ping I print pong!
        """
        print("pong!")
        