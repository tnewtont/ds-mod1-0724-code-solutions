#1
USE telecom;

#Conditional logic for volume_1
CREATE TEMPORARY TABLE dsstudent.nicetable
SELECT id, log_feature, volume,
	CASE
		WHEN volume < 100 THEN 'low'
		WHEN volume > 500 THEN 'large'
		ELSE 'medium'
	END volume_1
FROM log_feature AS lf;

	
# Selecting columns
SELECT *
FROM dsstudent.nicetable;

#Quantity of records for "low, "medium", and "Large
SELECT volume_1, COUNT(volume_1) AS value_counts
FROM dsstudent.nicetable
GROUP BY 1;

#2
#Conditional logic for HourlyRate_1
USE hr;

CREATE TEMPORARY TABLE dsstudent.rates
SELECT EmployeeNumber, HourlyRate,
	CASE
		WHEN HourlyRate >= 80 THEN 'high hourly rate'
		WHEN HourlyRate < 40 THEN 'low hourly rate'
		ELSE 'medium hourly rate'
	END HourlyRate_1
FROM hr.employee;

#Obtaining the columns
SELECT *
FROM dsstudent.rates;

#3
#Conditional logic for Gender_1
CREATE TEMPORARY TABLE dsstudent.genders
SELECT Gender,
	CASE
		WHEN Gender = 'Female' THEN 0
		ELSE 1
	END Gender_1
FROM hr.employee;

# Just to double check unique values within Gender column
SELECT COUNT(DISTINCT(Gender))
FROM hr.employee; # Returns 2

#Obtain columns
SELECT *
FROM dsstudent.genders;


	
	

	