# mi-ops

## File structure

```angular2html
mi-ops/
├── app/                      # Application layer, including UI and server logic
│   ├── __init__.py
│   ├── ui.py                 # User Interface logic
│   └── server.py             # Server-side logic for handling predictions and routing
│
├── models/                   # Model-related logic and operations
│   ├── load_model.py         # Directory for handling loading model from mlflow
│   └── model.py              # Training, saving, and processing models to mlflow
│
├── messaging/                # Messaging and communication logic (e.g., RabbitMQ-related code)
│   ├── __init__.py
│   └── commands.py           # Builds up RabbitMQ channels and connection
│
├── .gitignore                # Specifies files and directories to ignore in version control
├── README.md                 # Documentation about the project, including setup and usage instructions
└── requirements.txt          # Lists the Python dependencies required for the project
```
