from langchain.agents import create_agent
from langchain.tools import tool
import streamlit as st
import numpy as np
import dismech

@tool
def contortion(dt0: float, dt1: float, t_transition: float) -> str:
    """
    Simulate contortion with provided dt and get residual error of each time step.
    dt0 is the initial time step and dt1 is the time step after t'>t_transition.
    Returns 'error' if we fail to converge.
    """
    print(dt0, dt1, t_transition)
    
    # Move all initialization inside the function
    geom = dismech.GeomParams(rod_r0=0.001, shell_h=0)
    
    material = dismech.Material(
        density=1200, youngs_rod=2e6, youngs_shell=0, poisson_rod=0.5, poisson_shell=0
    )
    
    static_2d_sim = dismech.SimParams(
        static_sim=False,
        two_d_sim=False,
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=1,
        dt=dt0,  # Use the provided dt here
        max_iter=30,
        total_time=1.0,
        plot_step=1,
        tol=1e-4,
        ftol=1e-4,
        dtol=1e-2,
    )
    
    env = dismech.Environment()
    env.add_force("gravity", g=np.array([0.0, 0.0, -9.81]))
    
    geo = dismech.Geometry.from_txt("tests/resources/rod_cantilever/horizontal_rod_n21.txt")
    
    robot = dismech.SoftRobot(geom, material, geo, static_2d_sim, env)
    start = 0.01
    end = 0.09
    
    end_points = np.array(
        np.where(robot.state.q[robot.node_dof_indices].reshape(-1, 3)[:, 0] >= end)[0]
    )
    start_points = np.array(
        np.where(robot.state.q[robot.node_dof_indices].reshape(-1, 3)[:, 0] <= start)[0]
    )
    
    robot = robot.fix_nodes(np.union1d(start_points, end_points))
    
    def move_and_twist(robot: dismech.SoftRobot, t: float):
        """Simple example of a moving boundary condition"""
        u0 = 0.1
        w0 = 2
        
        if t < 0.15:
            robot = robot.move_nodes(start_points, [u0 * robot.sim_params.dt, 0, 0])
        else:
            robot = robot.twist_edges([0, 1], w0 * robot.sim_params.dt)
        if t > t_transition:
            robot.sim_params.dt = dt1
        return robot
    
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    stepper.before_step = move_and_twist
    
    # Run simulation
    try:
        _, _, f_norms = stepper.simulate()
    except ValueError:
        print(f"failed for dt0={dt0}, dt1={dt1}, t_transition={t_transition}")
        return "error"
    return str(f_norms)


# Initialize the agent once
@st.cache_resource
def get_agent():
    return create_agent(
        model="claude-sonnet-4-5-20250929",
        tools=[contortion],
        system_prompt="You are a helpful assistant. dt0=1e-2, dt1=1e-1, t=0.01 are good guesses. At max run 5 simulations before returning to the user.",
    )


agent = get_agent()

st.title("Contortion Experiment Runner")

query = st.text_area(
    "Enter your request:",
    "run contortion experiment with dt=1e-2 and return the lowest f_norm.",
    height=150,
)

if st.button("Run"):
    with st.spinner("Running agent..."):
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    st.subheader("Assistant Output")
    messages = result.get("messages", [])
    
    # Fix: Access message attributes directly, not with .get()
    assistant_msgs = [m for m in messages if hasattr(m, 'type') and m.type == "ai"]
    
    if assistant_msgs:
        for m in assistant_msgs:
            # Access content attribute directly
            st.write(m.content)
    else:
        st.write("No assistant messages found.")

    if "output" in result:
        st.subheader("Structured Output")
        st.json(result["output"])