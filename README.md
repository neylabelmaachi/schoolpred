<div align="center">
<h1>School Dropout Prediction 🎓</h1>
</div>

This tool has been built in order to prevent primary school dropout in high-risk rural regions of Morocco.
Student drop out can be identified as a pattern related to socio-economic backgrounds. It is a pressing issue in developing countries and it can be prevented when tackled early. Our analysis focus only on regios whre level of school dropout is high.

<div align="center">
<img width="490" alt="image" src="https://github.com/neylabelmaachi/schoolpred/assets/120349975/785d4bf1-c434-425a-9290-8bd8d94110f0">
</div>

## Why do we care?
* Higher unemployment rate; 
* increased poverty; 
* out-of-school students;
* increased violence

Rural environment are often neglected by the government, the tool we created allows to raise awareness and easily identify at-risk students. 

[Link to the app on Streamlit](https://lewagon-schooldropout.streamlit.app/)


<div align="center">
<h1>DB Schema</h1>
</div>

<img width="748" alt="Screenshot 2023-05-29 at 10 48 38" src="https://github.com/neylabelmaachi/schoolpred/assets/37574368/9de5e612-48f2-4d28-8173-f568719d9338">


<div align="center">
<h1>Key columns</h1>
</div>

**baseline_household**
 - `work_activity` - Work status of the parents
 - `individual_water_net` - Individual water network connection
 - `electrical_net` - Electric connection
 - `mobile_phones` - If the family have a mobile phone or not
 - `type_housing` - The architectural structure of the house 

**child_math_test_result**
 - `digit_recognition_res` - Digit recognition test results
 - `number_recognition` - Number recognition test results
 - `subtraction_res` - Subtraction test results
 - `division_res` - Division test resuls
 - `average_math_score` - Average score of the 4 math test sections mentioned above 


<div align="center">
<h1>Reference Key</h1>
</div>

| Variable Name   | Encoded Numbers | Description |
|-----------------|------|-------------|
| mother_alive    | 1    | Yes         |
|                 | 2    | No          |
| father_alive    | 1    | Yes         |
|                 | 2    | No          |
| parents_age     | -    | Age in Years|
| marital_status  | 1    | Married     |
|                 | 2    | Single      |
|                 | 3    | Divorced    |
|                 | 4    | Widowed     |
| parents_level_ed| 1    | No education|
|                 | 2    | Religious education|
|                 | 3    | Primary School|
|                 | 4    | Middle School|
|                 | 5    | High School|
|                 | 6    | Higher Education|
|                 | 7    | Professional Training|
| work_activity   | 1    | Full Time|
|                 | 2    | Part Time|
|                 | 3    | Unemployed|
| type_housing    | 1    | Adobe/Clay house|
|                 | 2    | Permanent House|
|                 | 3    | Dry Stone|
|                 | 4    | Modern/Concrete house|
|                 | 5    | Other|

<div align="center">
<h1>Dataset</h1>
</div>

Our prediction model has been trained on the following research dataset: Data for Development Initiative. (2019). Morocco CCT Education (Version 1.0) 
[Data set](https://redivis.com/datasets/11xy-bb1z6q7ap?v=1.0)
