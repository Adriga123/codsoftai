def chatbot():
    print("ChatBot: Hi! I'm ChatBot. Ask me anything or type 'bye' to exit.")

    while True:
        try:
            user_input = input("You: ").lower().strip()
            
            if user_input in ["hi","hello","hey","greetings"]:
                print("Chatbot: HELLO !How can i assist you today?")
            elif "your name" in user_input or "who are you" in user_input:
                print("Chatbot: HI,I AM A SIMPLE RULE BASED CHATBOT")
            elif "how are you" in user_input or "how do you do" in user_input:
                print("Chatbot: I'm Fine and doing well,thank you! what about you?")
            elif "time" in user_input:
                from datetime import datetime
                now = datetime.now().strftime("%H:%M:%S")
                print(f"Chatbot: The current time is{now}.")
            elif "date" in user_input:
                from datetime import date
                today = datetime.today().strftime("%d,%m,%Y")
                print(f"Chatbot:Today's date is{today}.")
            elif "help" in user_input or "options" in user_input:
                print("Chatbot: I can tell you the date, time greet you and share my name")
            elif user_input in ["bye","exit","quit","goodbye"]:
                print("Chatbot: GOODBYE!,Have a nice day!")
                break
            else:
                print("Chatbot:I'm sorry, i don't understand that.Try something else.")
                

            
        except KeyboardInterrupt:
            print("\nChatBot: Goodbye! Have a great day!")
            break
        except Exception as e:
            print(f"ChatBot: Sorry, something went wrong. ({str(e)})")

chatbot()