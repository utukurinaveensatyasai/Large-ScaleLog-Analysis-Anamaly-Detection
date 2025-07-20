# log_analysis_and_anomaly_detection.py
import os
import random
import time
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, hour, count, avg, stddev, from_unixtime, unix_timestamp, countDistinct
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, LongType

print("Starting Large-Scale Log Analysis and Anomaly Detection Project...")

# --- Configuration ---
LOG_FILE_PATH = "synthetic_logs.log"
NUM_LOG_ENTRIES = 100000 # Number of log entries to generate
LOG_GENERATION_PERIOD_DAYS = 30 # Logs span 30 days
ANOMALY_THRESHOLD_FACTOR = 3.0 # Factor for standard deviation for anomaly detection

# --- 1. Synthetic Log Data Generation ---
# This function creates a large, realistic-looking log file.
# It simulates different log levels, services, messages, and timestamps.
def generate_synthetic_logs(file_path, num_entries, period_days):
    """
    Generates a synthetic log file with a specified number of entries.

    Args:
        file_path (str): The path to save the generated log file.
        num_entries (int): The total number of log entries to generate.
        period_days (int): The number of days over which logs are distributed.
    """
    print(f"Generating {num_entries} synthetic log entries to {file_path}...")
    log_levels = ["INFO", "WARN", "ERROR", "DEBUG", "CRITICAL"]
    services = ["UserService", "ProductService", "OrderService", "PaymentGateway", "AuthService", "InventoryService"]
    info_messages = [
        "User logged in successfully.",
        "Data fetched from database.",
        "Processing request for user.",
        "Cache refreshed.",
        "API call successful."
    ]
    warn_messages = [
        "Deprecated API usage detected.",
        "High latency detected for external service.",
        "Resource limit approaching.",
        "Configuration mismatch."
    ]
    error_messages = [
        "Database connection failed.",
        "NullPointerException occurred.",
        "Service unavailable.",
        "Authentication failed.",
        "Disk space low."
    ]
    critical_messages = [
        "System shutdown initiated.",
        "Critical security vulnerability detected.",
        "Data corruption detected."
    ]

    start_time = datetime.now() - timedelta(days=period_days)
    end_time = datetime.now()

    with open(file_path, "w") as f:
        for i in range(num_entries):
            # Distribute timestamps evenly over the period
            timestamp = start_time + (end_time - start_time) * (i / num_entries)
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            level = random.choice(log_levels)
            service = random.choice(services)

            if level == "INFO":
                message = random.choice(info_messages)
            elif level == "WARN":
                message = random.choice(warn_messages)
            elif level == "ERROR":
                message = random.choice(error_messages)
            elif level == "CRITICAL":
                message = random.choice(critical_messages)
            else: # DEBUG
                message = f"Debug message for {service} operation."

            # Introduce some anomalies (e.g., sudden spike in CRITICAL errors)
            if random.random() < 0.001: # 0.1% chance for a specific anomaly
                level = "CRITICAL"
                message = "UNEXPECTED_SYSTEM_FAILURE_CORE_DUMP"
            elif random.random() < 0.005 and level == "ERROR": # 0.5% chance for a specific error pattern
                message = "ERROR: Too many failed login attempts from IP 192.168.1.X"

            log_entry = f"{timestamp_str} [{level}] [{service}] {message}\n"
            f.write(log_entry)
    print(f"Log generation complete. File saved to {file_path}")

# Generate the log file
generate_synthetic_logs(LOG_FILE_PATH, NUM_LOG_ENTRIES, LOG_GENERATION_PERIOD_DAYS)

# --- 2. Initialize Spark Session ---
# SparkSession is the entry point to programming Spark with the Dataset and DataFrame API.
print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("LogAnalysis") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

print("Spark Session initialized.")

# --- 3. Define Schema and Load Data ---
# Defining a schema helps Spark parse the data efficiently and correctly.
print(f"Loading log data from {LOG_FILE_PATH} into Spark DataFrame...")
log_schema = StructType([
    StructField("timestamp_str", StringType(), True),
    StructField("level", StringType(), True),
    StructField("service", StringType(), True),
    StructField("message", StringType(), True)
])

# Read the log file line by line and parse it
# We use RDDs for parsing complex lines, then convert to DataFrame
logs_rdd = spark.sparkContext.textFile(LOG_FILE_PATH)

# Parse each log line using regex or string splitting
# Example log format: "YYYY-MM-DD HH:MM:SS [LEVEL] [SERVICE] Message"
parsed_logs_rdd = logs_rdd.map(lambda line: line.split(' ', 3)) \
                          .filter(lambda parts: len(parts) == 4) \
                          .map(lambda parts: (
                              parts[0] + ' ' + parts[1], # timestamp_str
                              parts[2].strip('[]'),      # level
                              parts[3].strip('[]'),      # service
                              parts[3]                   # message (original full message)
                          ))

# Convert RDD to DataFrame
logs_df = spark.createDataFrame(parsed_logs_rdd, log_schema)

# Further parsing and cleaning using DataFrame operations
logs_df = logs_df.withColumn("timestamp", from_unixtime(unix_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss"))) \
                 .drop("timestamp_str") # Drop the original string timestamp column

logs_df.printSchema()
logs_df.show(5, truncate=False)
print(f"Total log entries loaded: {logs_df.count()}")

# --- 4. Exploratory Data Analysis (EDA) ---
print("\n--- Exploratory Data Analysis ---")

# Count by Log Level
print("\nLog Entry Counts by Level:")
logs_df.groupBy("level").count().orderBy(col("count").desc()).show(truncate=False)

# Count by Service
print("\nLog Entry Counts by Service:")
logs_df.groupBy("service").count().orderBy(col("count").desc()).show(truncate=False)

# Top Error Messages
print("\nTop 10 Error Messages:")
logs_df.filter(col("level") == "ERROR") \
       .groupBy("message").count().orderBy(col("count").desc()).limit(10).show(truncate=False)

# Log trends over time (hourly)
print("\nHourly Log Counts (last 24 hours of data):")
logs_df.withColumn("log_hour", hour(col("timestamp"))) \
       .filter(col("timestamp") >= (datetime.now() - timedelta(hours=24))) \
       .groupBy("log_hour").count().orderBy("log_hour").show()

# --- 5. Anomaly Detection: Spike in Error/Critical Logs ---
# We'll detect anomalies based on the count of ERROR/CRITICAL logs per hour.
print("\n--- Anomaly Detection: Spike in Error/Critical Logs ---")

# Aggregate error/critical logs by hour
hourly_error_counts = logs_df.filter((col("level") == "ERROR") | (col("level") == "CRITICAL")) \
                             .withColumn("log_hour_ts", from_unixtime(unix_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"), "yyyy-MM-dd HH")) \
                             .groupBy("log_hour_ts") \
                             .agg(count("*").alias("error_count")) \
                             .orderBy("log_hour_ts")

# Calculate mean and standard deviation of error counts
stats = hourly_error_counts.agg(
    avg("error_count").alias("mean_error_count"),
    stddev("error_count").alias("stddev_error_count")
).collect()[0]

mean_error_count = stats["mean_error_count"]
stddev_error_count = stats["stddev_error_count"]

print(f"Mean hourly error/critical count: {mean_error_count:.2f}")
print(f"Standard deviation of hourly error/critical count: {stddev_error_count:.2f}")

# Define anomaly threshold
anomaly_threshold = mean_error_count + ANOMALY_THRESHOLD_FACTOR * stddev_error_count
print(f"Anomaly threshold (Mean + {ANOMALY_THRESHOLD_FACTOR}*StdDev): {anomaly_threshold:.2f}")

# Identify anomalous hours
anomalous_hours = hourly_error_counts.filter(col("error_count") > anomaly_threshold)

print("\nIdentified Anomalous Hours (Spike in Errors/Criticals):")
if anomalous_hours.count() > 0:
    anomalous_hours.show(truncate=False)
    # To investigate specific logs during anomalous periods:
    print("\nSample logs from an anomalous period (e.g., first detected anomalous hour):")
    first_anomalous_hour_str = anomalous_hours.select("log_hour_ts").first()["log_hour_ts"]
    if first_anomalous_hour_str:
        start_of_hour = datetime.strptime(first_anomalous_hour_str, "%Y-%m-%d %H")
        end_of_hour = start_of_hour + timedelta(hours=1)
        logs_df.filter((col("timestamp") >= start_of_hour) & (col("timestamp") < end_of_hour)) \
               .filter((col("level") == "ERROR") | (col("level") == "CRITICAL")) \
               .limit(10).show(truncate=False)
else:
    print("No significant anomalies detected based on the defined threshold.")

# --- 6. Anomaly Detection: Rare Events / Specific Message Patterns ---
# Example: Detect specific critical messages that are very rare but indicate severe issues.
print("\n--- Anomaly Detection: Rare Critical Messages ---")
rare_critical_message = "UNEXPECTED_SYSTEM_FAILURE_CORE_DUMP"
rare_events_df = logs_df.filter(col("message").contains(rare_critical_message))

print(f"Detected occurrences of '{rare_critical_message}':")
if rare_events_df.count() > 0:
    rare_events_df.show(truncate=False)
else:
    print(f"No occurrences of '{rare_critical_message}' detected.")


# --- 7. Reporting and Recommendations ---
print("\n--- Project Report and Recommendations ---")
print("This analysis processed system logs to identify patterns and potential anomalies.")
print(f"Total log entries analyzed: {logs_df.count()}")
print(f"Log data spanned: {LOG_GENERATION_PERIOD_DAYS} days")

print("\n**Key Findings:**")
print("- Identified common log levels and service activity patterns.")
print("- Top error messages provide insights into recurring software issues.")
if anomalous_hours.count() > 0:
    print(f"- Detected {anomalous_hours.count()} anomalous hourly spikes in ERROR/CRITICAL logs, indicating potential system instability or attack attempts.")
else:
    print("- No significant hourly error/critical log spikes detected, indicating stable operation.")

if rare_events_df.count() > 0:
    print(f"- Identified {rare_events_df.count()} occurrences of critical system failures ('{rare_critical_message}'), which require immediate attention.")
else:
    print(f"- No instances of '{rare_critical_message}' were found.")

print("\n**Recommendations for System Improvement & Security Enhancements:**")
print("1.  **Automated Alerting:** Implement real-time alerting for detected anomalies (e.g., hourly error spikes) and critical messages. Integrate with paging/notification systems.")
print("2.  **Root Cause Analysis:** For each identified anomalous period or critical message, conduct a deep dive to understand the underlying cause (e.g., code deployment, infrastructure failure, malicious activity).")
print("3.  **Log Retention Policy:** Ensure appropriate log retention policies are in place for historical analysis and compliance.")
print("4.  **Threshold Tuning:** Continuously monitor and fine-tune anomaly detection thresholds (e.g., ANOMALY_THRESHOLD_FACTOR) based on system behavior and business context to minimize false positives/negatives.")
print("5.  **Security Monitoring:** For patterns like 'failed login attempts', integrate with security information and event management (SIEM) systems for broader threat detection.")
print("6.  **Performance Optimization:** Analyze log messages related to high latency or resource limits to identify and resolve performance bottlenecks in services.")
print("7.  **Dashboarding:** Develop a real-time dashboard (e.g., using Grafana, Kibana) to visualize log metrics and anomalies for operational teams.")

print("\nProject execution complete. Review the findings and recommendations.")

# Stop Spark Session
spark.stop()
print("Spark Session stopped.")
