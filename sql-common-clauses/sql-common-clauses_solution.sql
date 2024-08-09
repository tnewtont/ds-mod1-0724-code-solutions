#SELECT `SELECT` to bypass an in-built feature (use backticks)
# Note: I rewrote the SQL queries based on the instructions written, not based on the output pictures

#1 SELECT FROM/AS

USE telecom;

SELECT id, log_feature AS `log`, volume AS `vol`
from log_feature;

#Sorting
#1 Write a SQL query to return the first 5 rows of “id”, “resource_type” and sorted by ”id” column in ascending order.
SELECT id, resource_type
from resource_type
ORDER BY 1
LIMIT 5;

#2 Write a SQL query to return the last 5 rows of “id”, “resource_type” and sorted by ”id” column in descending order.
SELECT id, resource_type
from resource_type
ORDER BY 1 DESC
LIMIT 5;

#3 Write a SQL query to return 5 rows of “id”, “resource_type” and sorted by ”id” column in ascending order first,
# then sorted by “resource_type” column in a descending order.
SELECT id, resource_type
from resource_type
ORDER BY 1 DESC, 2
LIMIT 5;

#Count/distinct
#1 Write a SQL query to return the following data from severity_type:
# Numbers of rows
# Numbers of unique values of column ‘id’
# Numbers of unique values of column ‘severity_type’
SELECT COUNT(*) AS numbers_row, COUNT(DISTINCT(id)) AS id_nunique, COUNT(DISTINCT(severity_type)) AS severity_type_unique
FROM severity_type;

# WHERE needs to come after FROM, is is only for special keywords

# WHERE filtering
#1 Write a SQL query to return from the “log_feature” table, ”feature_201” with a volume between 100 and 300.
# In the query result, return ‘id’, ‘log_feature’, ‘volume’ columns only
# Sort the result by the ‘volume’ column 
SELECT id, log_feature, volume
FROM log_feature
WHERE volume BETWEEN 100 AND 300 AND log_feature = 'feature 201' 
ORDER BY 3 ASC;

# WHERE ID in (num1, num2, etc)