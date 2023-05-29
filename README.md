# School Dropout Predict

## DB Schema
<img width="748" alt="Screenshot 2023-05-29 at 10 48 38" src="https://github.com/neylabelmaachi/schoolpred/assets/37574368/9de5e612-48f2-4d28-8173-f568719d9338">

### Key columns
**baseline_household**
 - `work_activity` - Work status of the parents
 - `individual_water_net` - Individual water network connection
 - `electrical_net` - Electric connection
 - `mobile_phones` - If the family have a mobile phone or not
 - 'type_housing' - The architectural structure of the house 

**child_math_test_result**
 - `digit_recognition_res` - Digit recognition test results
 - `number_recognition` - Number recognition test results
 - `subtraction_res` - Subtraction test results
 - `division_res` - Division test resuls
 - `average_math_score` - Average score of the 4 math test sections mentioned above 


<center>
<h1>Reference Key</h1>
</center>

### Reference Key 
**features**
- `mother_alive` - {1:Yes, 2:No}
- `father_alive` - {1:Yes, 2:No}
- `parents_age` in years
- `marital_status` - {1:Married, 2:Single, 3:Divorced, 4:Widowed}
- `parents_level_ed` - {1:No education, 2:Religious, 3:Primary School, 4:Middle School, 5:High School, 6:Higher Education, 7:Professional Training}
- `work_activity` - {1:Full Time, 2:Part Time, 3:Unemployed}
- `type_housing` -  {1:Adobe/Clay house, 2:Permanent House, 3:Dry Stone, 4:Modern/Concrete house, 5:Other}
