#SELECT `SELECT` to bypass an in-built feature (use backticks)


#1

USE telecom;

SELECT id, log_feature AS `log`, volume AS `vol`
from log_feature;

#Sorting
#2
SELECT id, resource_type
from resource_type
ORDER BY 1 ASC
LIMIT 5;

#3
SELECT id, resource_type
from resource_type
ORDER BY 1, 2 ASC
LIMIT 5;

#4
SELECT id, resource_type
from resource_type
ORDER BY 1 DESC
LIMIT 5;

#Count/distinct
SELECT COUNT(*) AS numbers_row, COUNT(id) AS id_nunique, COUNT(DISTINCT(severity_type)) AS severity_type_unique
FROM severity_type;

# WHERE needs to come after FROM, is is only for special keywords

# WHERE filtering
#6 
SELECT id, `log_feature`, volume
FROM log_feature
WHERE volume BETWEEN 100 AND 300 AND log_feature = 'feature 201' 
ORDER BY 3 ASC;
# WHERE ID in (num1, num2, etc)