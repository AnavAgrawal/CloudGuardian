import os

# Create directories for the project
os.makedirs('cloud_guardian/models', exist_ok=True)
os.makedirs('cloud_guardian/api', exist_ok=True)
os.makedirs('cloud_guardian/dashboard', exist_ok=True)

# Create empty files for main components
open('cloud_guardian/main.py', 'w').close()
open('cloud_guardian/models/resource_optimizer.py', 'w').close()
open('cloud_guardian/models/security_monitor.py', 'w').close()
open('cloud_guardian/api/endpoints.py', 'w').close()
open('cloud_guardian/dashboard/dashboard.py', 'w').close()

print("Project structure created successfully.")
