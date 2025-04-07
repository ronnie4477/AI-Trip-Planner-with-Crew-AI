import streamlit as st
from datetime import datetime, timedelta, date
import sys

from crewai import Crew, LLM
from trip_agents import TripAgents, StreamToExpander
from trip_tasks import TripTasks


st.set_page_config(page_title="AITravelPlanner ✈️", page_icon="✈️", layout="wide")


def show_emoji_icon(emoji: str):
    st.markdown(f"<div style='font-size: 72px;'>{emoji}</div>", unsafe_allow_html=True)


class TripCrew:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = f"{date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}"
        self.interests = interests
        self.output_placeholder = st.empty()
        self.llm = LLM(model="gemini/gemini-2.0-flash")
        # Uncomment if using OpenAI
        # self.llm = OpenAI(temperature=0.7, model_name="gpt-4")

    def run(self):
        try:
            agents = TripAgents(llm=self.llm)
            tasks = TripTasks()

            city_selector = agents.city_selection_agent()
            local_expert = agents.local_expert()
            concierge = agents.travel_concierge()

            identify = tasks.identify_task(city_selector, self.origin, self.cities, self.interests, self.date_range)
            gather = tasks.gather_task(local_expert, self.origin, self.interests, self.date_range)
            plan = tasks.plan_task(concierge, self.origin, self.interests, self.date_range)

            crew = Crew(agents=[city_selector, local_expert, concierge],
                        tasks=[identify, gather, plan],
                        verbose=True)

            result = crew.kickoff()
            self.output_placeholder.markdown(result)
            return result
        except Exception as e:
            st.error(f"🚨 Oops! Something went wrong: {str(e)}")
            return None


# --- Main App ---
def main():
    show_emoji_icon("🚀🌄")
    st.title("Trippy Tales: Your AI Travel Planner")

    st.markdown("Your personal AI assistant is ready to create the ultimate trip plan based on your preferences.")

    today = date.today()
    default_end_date = date(today.year + 1, 1, 16)

    with st.sidebar:
        st.header("🧳 Trip Preferences")
        with st.form("trip_form"):
            origin = st.text_input("🌍 Your Current Location", placeholder="San Mateo, CA")
            destination = st.text_input("📍 Desired Destination", placeholder="Bali, Indonesia")
            date_range = st.date_input("📅 Travel Dates", value=(today, default_end_date), min_value=today)
            interests = st.text_area("🧠 Interests & Details",
                                     placeholder="2 adults who love swimming, hiking, food...")

            submitted = st.form_submit_button("✈️ Plan My Trip")

        st.markdown("---")
        

        

    if submitted:
        st.divider()
        with st.status("🔍 Gathering your trip details...", expanded=True) as status:
            with st.container(height=450):
                sys.stdout = StreamToExpander(st)
                trip = TripCrew(origin, destination, date_range, interests)
                result = trip.run()

            status.update(label="✅ Your trip plan is ready!", state="complete")

        st.divider()
        st.subheader("🗺️ Here’s your personalized travel plan", divider="rainbow")
        st.markdown(result)


if __name__ == "__main__":
    main()
