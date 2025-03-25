"""                                                                                                                                                                                                                                    
config.py                                                                                                                                                                                                                              
                                                                                                                                                                                                                                       
This module provides a configuration management system for the application.                                                                                                                                                            
It loads configuration settings from a YAML file and allows access to these settings                                                                                                                                                   
through a singleton instance of the Config class.                                                                                                                                                                                      
                                                                                                                                                                                                                                       
Usage Example:                                                                                                                                                                                                                         
                                                                                                                                                                                                                                       
1. Import the config instance:                                                                                                                                                                                                         
   from src.config import config                                                                                                                                                                                                       
                                                                                                                                                                                                                                       
2. Access a configuration value:                                                                                                                                                                                                       
   api_key = config.get("openai.api_key")                                                                                                                                                                                              
"""     
import os
import re
import yaml
from pathlib import Path

class Config:
    # This class manages application configuration, including loading domain-specific keywords.
    """                                                                                                                                                                                                                                
    A class to manage application configuration.                                                                                                                                                                                       
                                                                                                                                                                                                                                       
    This class loads configuration settings from a specified YAML file,                                                                                                                                                                
    expands environment variables, and provides methods to access the configuration.                                                                                                                                                   
                                                                                                                                                                                                                                       
    Attributes:                                                                                                                                                                                                                        
        config_path (Path): The path to the configuration file.                                                                                                                                                                        
        config (dict): The loaded configuration settings.                                                                                                                                                                              
                                                                                                                                                                                                                                       
    Parameters:                                                                                                                                                                                                                        
        config_path (str): The path to the configuration file (default is "config.yaml").                                                                                                                                              
    """
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(__file__).parent.parent / config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """                                                                                                                                                                                                                            
        Load the configuration from the specified YAML file.                                                                                                                                                                           
                                                                                                                                                                                                                                       
        This method reads the configuration file, expands any environment variables                                                                                                                                                    
        present in the file, and returns the configuration as a dictionary.                                                                                                                                                            
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            dict: The loaded configuration settings.                                                                                                                                                                                   
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            FileNotFoundError: If the configuration file does not exist.                                                                                                                                                               
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Expand environment variables of the form ${VAR_NAME}
        content = re.sub(
            r"\$\{([^}]+)\}",
            lambda match: os.environ.get(match.group(1), match.group(0)),  # fallback to placeholder if not found
            content
        )

        return yaml.safe_load(content)

    def get(self, key, default=None):
        """                                                                                                                                                                                                                            
        Retrieve a configuration value by key.                                                                                                                                                                                         
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            key (str): The key of the configuration setting to retrieve.                                                                                                                                                               
            default: The default value to return if the key is not found (default is None).                                                                                                                                            
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            The value of the configuration setting, or the default value if the key is not found.                                                                                                                                      
        """
        return self.config.get(key, default)

    def __getitem__(self, key):
        """                                                                                                                                                                                                                            
        Retrieve a configuration value using the indexing syntax.                                                                                                                                                                      
                                                                                                                                                                                                                                       
        Parameters:                                                                                                                                                                                                                    
            key (str): The key of the configuration setting to retrieve.                                                                                                                                                               
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            The value of the configuration setting.                                                                                                                                                                                    
                                                                                                                                                                                                                                       
        Raises:                                                                                                                                                                                                                        
            KeyError: If the key is not found in the configuration.                                                                                                                                                                    
        """
        return self.config[key]

    def __repr__(self):
        """                                                                                                                                                                                                                            
        Return a string representation of the Config object.                                                                                                                                                                           
                                                                                                                                                                                                                                       
        Returns:                                                                                                                                                                                                                       
            str: A string representation of the Config object, including the path to the config file.                                                                                                                                  
        """
        return f"Config({self.config_path})"


# Singleton pattern for configuration across the application
config = Config()
