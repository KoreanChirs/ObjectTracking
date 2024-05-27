import random
FPS = 25
dt = 1/FPS

def physics_based_prediction(arr,dt):
    positions = arr

    velocity = [0]
    #velocity[i] = (position[i] - position[i-1])/dt
    for i in range(1,len(positions)):
        velocity.append((positions[i] - positions[i-1])/dt)

    accel = [0,0]
    for i in range(2,len(velocity)):
        accel.append((velocity[i] - velocity[i-1])/dt)

    jerk = [0,0,0]
    for i in range(3,len(velocity)):
        jerk.append((accel[i] - accel[i-1])/dt)
    
    # predict position
    predictions = [None]*len(positions)

    # for idx 0
    if (len(positions) > 0):
        predictions[0] = None
    #for idx 1
    if (len(positions) > 1):
        predictions[1] = positions[0]
    #for idx 2
    if (len(positions) > 2):
        predictions[2] = positions[1] + velocity[1]*dt #원래는 velocity[2]가지고 해야하는데, 방법이 없음
    #for idx 3
    if (len(positions) > 3):
        velocity_predict = velocity[2] + accel[2]*dt
        predictions[3] = positions[2] + velocity_predict*dt
    for idx in range(4,len(positions)):
        acc_tmp = accel[idx-1] + jerk[idx-1]*dt
        vel_tmp = velocity[idx-1] + acc_tmp*dt
        predictions[idx] = positions[idx-1] + vel_tmp*dt
    
    return predictions


#for test_phy.py
def physics_based_prediction_2(arr,dt):
    positions = arr

    velocity = [0]
    #velocity[i] = (position[i] - position[i-1])/dt
    for i in range(1,len(positions)):
        velocity.append((positions[i] - positions[i-1])/dt)

    accel = [0,0]
    for i in range(2,len(velocity)):
        accel.append((velocity[i] - velocity[i-1])/dt)

    jerk = [0,0,0]
    for i in range(3,len(velocity)):
        jerk.append((accel[i] - accel[i-1])/dt)
    
    # predict position

    #for idx 1
    if (len(positions) == 1):
        predictions = positions[0]
    #for idx 2
    if (len(positions) == 2):
        predictions = positions[1] + velocity[1]*dt #원래는 velocity[2]가지고 해야하는데, 방법이 없음
    #for idx 3
    if (len(positions) == 3):
        velocity_predict = velocity[2] + accel[2]*dt
        predictions = positions[2] + velocity_predict*dt
    
    return predictions


if __name__ == "main":
    sample_arr = [0.5*3*i**2 for i in range(10)]
    random_arr = [random.randint(0,100) for _ in range(10)]
    print(sample_arr)
    print(physics_based_prediction(sample_arr,dt))


