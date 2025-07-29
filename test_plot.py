from plot_trajectory import plot_flight, plot_multiple_flights

# 1. Plot a specific flight in both 2D & 3D and save
print("Plotting specific flight (100001)...")
plot_flight(flight_id="100001", both=True, save=True)

# 2. Plot a random flight in 2D and save
print("Plotting random flight (2D)...")
plot_flight(mode="2D", save=True)

# 3. Multi-flight comparison (3 random flights)
print("Plotting multi-flight comparison...")
plot_multiple_flights(count=3, save=True)

print("All plots saved in 'plots/' directory.")