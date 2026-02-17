import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Hardcoded data for Seorinarayan
dates_str = [
    '1993-06-15', '1993-06-16', '1993-06-17', '1993-06-18', '1993-06-19', 
    '1993-06-20', '1993-06-21', '1993-06-22', '1993-06-23', '1993-06-24', 
    '1993-06-25', '1993-06-26', '1993-06-27', '1993-06-28', '1993-06-29', 
    '1993-06-30', '1993-07-01', '1993-07-02', '1993-07-03', '1993-07-04', 
    '1993-07-05', '1993-07-06', '1993-07-07', '1993-07-08', '1993-07-09', 
    '1993-07-10', '1993-07-11', '1993-07-12', '1993-07-13', '1993-07-14', 
    '1993-07-15'
]

observed_values = [
    3.167, 4.225, 4.675, 4.558, 9.275, 18.93, 36.0, 55.53, 53.58, 34.58, 
    44.63, 46.78, 80.61, 106.2, 116.8, 115.8, 113.0, 105.6, 92.25, 95.0, 
    88.54, 89.18, 164.6, 372.0, 1464.0, 1269.0, 658.6, 407.1, 302.2, 404.4, 
    1707.0
]

imputed_values = [
    22.97, 49.99, 128.91, 108.17, 118.43, 110.57, 134.45, 133.15, 128.44, 
    124.48, 165.94, 182.74, 203.62, 201.96, 170.22, 162.48, 156.64, 159.82, 
    144.66, 138.77, 141.21, 142.30, 216.21, 1396.73, 1334.37, 1148.58, 
    630.62, 534.64, 733.84, 866.06, 1707.0
]

# Convert date strings to datetime objects
dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

# Create the plot
plt.figure(figsize=(10, 5))

# Plot Observed data
plt.plot(dates, observed_values, label='Observed', color='dimgray')

# Plot Imputed data
plt.plot(dates, imputed_values, label='Imputed', color='red', marker='.', linestyle='--')

# Styling
plt.title('seorinarayan | Gap: 30d | ID: 1')
plt.xlabel('Date')
plt.ylabel('Discharge')
plt.legend()
plt.grid(True, alpha=0.3)

# Configure x-axis ticks to prevent overlap
# minticks=3, maxticks=6 ensures we don't crowd the axis
locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
formatter = mdates.DateFormatter('%Y-%m-%d')

ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

# Automatically rotate and align labels
plt.gcf().autofmt_xdate()

# Show the plot
plt.tight_layout()
plt.show()