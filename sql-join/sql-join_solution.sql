USE telecom;

SELECT t.id, t.location, t.fault_severity, et.event_type, st.severity_type, rt.resource_type, lf.log_feature, lf.volume
FROM train AS t LEFT OUTER JOIN event_type as et ON t.id = et.id
				LEFT OUTER JOIN severity_type as st ON t.id = st.id
				LEFT OUTER JOIN resource_type as rt ON t.id = rt.id
				LEFT OUTER JOIN log_feature as lf ON t.id = lf.id;

