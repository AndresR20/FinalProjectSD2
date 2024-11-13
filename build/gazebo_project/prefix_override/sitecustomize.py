import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/andres/Documents/autonomous-vehicle-end-to-end-control-project-template/install/gazebo_project'
