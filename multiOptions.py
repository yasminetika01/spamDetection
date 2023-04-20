import streamlit as st

# Define the multioption class to manage the multiple apps in our program 
class Multioption: 
    def __init__(self) -> None:
        self.options = []
    
    def add_option(self, title, func) -> None: 
       
        self.options.append(
            {
                "title": title, 
                "function": func
            }
        )

    def run(self):
        # Drodown to select the option to run  
        option = st.sidebar.selectbox(
            'Options', 
            self.options, 
            format_func=lambda option: option['title']
        )

        # run the app function 
        option['function']()