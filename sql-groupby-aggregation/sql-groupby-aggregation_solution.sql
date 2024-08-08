# First part is telecom, second part is hr

#1 This is a continuation from the sql-join assignment, so make a temp table first
# Q: For each location, what is the quantity of unique event types?
USE telecom;

CREATE TEMPORARY TABLE dsstudent.supertable
SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
FROM train AS t LEFT OUTER JOIN event_type as et ON t.id = et.id
				LEFT OUTER JOIN severity_type as st ON t.id = st.id
				LEFT OUTER JOIN resource_type as rt ON t.id = rt.id
				LEFT OUTER JOIN log_feature as lf ON t.id = lf.id;

SELECT location, COUNT(DISTINCT(event_type)) AS num_unique_event_type
FROM dsstudent.supertable
GROUP BY Location;

#Q What are the top 3 locations with the most volumes?
SELECT location, SUM(volume) AS total_volume
FROM dsstudent.supertable
GROUP BY 1
ORDER BY 2 DESC
LIMIT 3;

#2
#Q: For each fault severity, what is the quantity of unique locations?
SELECT fault_severity, COUNT(DISTINCT(location)) AS num_of_unique_locations
FROM dsstudent.supertable
GROUP BY fault_severity;

#Q: From the query result above, what is the quantity of unique locations with the fault_severity greater than 1?
SELECT fault_severity, COUNT(DISTINCT(location)) AS num_of_unique_locations
FROM dsstudent.supertable
GROUP BY fault_severity
HAVING fault_severity > 1;

#3
#Q: Write a SQL query to return the minimum, maximum, average of the field “Age” for each “Attrition” groups from the “hr” database
USE hr;

SELECT Attrition, MIN(Age) AS min_age, MAX(Age) AS max_age, AVG(Age) AS avg_age
FROM employee
GROUP BY Attrition;

#4
#Q:Write a SQL query to return the “Attrition”, “Department” and the number of records from the ”hr” database for each group in the “Attrition” and “Department.”
# Sort the returned table by the “Attrition” and “Department” fields in ascending order.

SELECT Attrition, Department, COUNT(*) AS num_quantity
FROM employee
GROUP BY 1, 2
HAVING Attrition IS NOT NULL
ORDER BY 1, 2;

#5
#Q: From Question #4, can you return the results where the “num_quantity” is greater than 100 records?

SELECT Attrition, Department, COUNT(*) AS num_quantity
FROM employee
GROUP BY 1, 2
HAVING Attrition IS NOT NULL AND num_quantity > 100
ORDER BY 1, 2;
