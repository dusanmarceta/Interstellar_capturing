import numpy as np
import rebound
from tqdm import tqdm
import time
import sys
import random
# generator
from synthetic_population_shell import synthetic_population_shell
from auxiliary_functions import ecc2true

# ja sam bez ovoga jednom imao neki error
import rebound.horizons
rebound.horizons.SSL_CONTEXT = 'unverified'

core_id = sys.argv[1]
stellar_type = ""
if 1 <= int(core_id) <= 12:
    stellar_type = "M"
elif 13 <= int(core_id) <= 24:
    stellar_type = "G"
elif 25 <= int(core_id) <= 36:
    stellar_type = "OB"

TIME_PER_BATCH_YEARS = 2500
STEP_YEARS = 1.0 / 365.0 
CAPTURE_CHECK_INTERVAL_STEPS = 100
CAPTURE_LOG_FILE = f'captures_core{core_id}_{stellar_type}.txt'
PROGRESS_LOG_FILE = f'progress_core{core_id}_{stellar_type}.txt'

# Za svaku integraciju biramo epohu izmedju epoha_0 i epoha_0 + max_dana
epoha_0 = 2456200.5000
max_dana = 2000 * 365.25

# Shell u kom definisemo objekte
r_min = 1000
r_max = 1001


# parametri za generator
#R_MODEL = 50 
N0 = 1e-1
V_MIN = 1e3
V_MAX = 2e5
U_SUN = 1e4
V_SUN = 1.1e4
W_SUN = 7e3
#SIGMA_VX = 1.2e4 # ovo menjamo dole
#SIGMA_VY = 1.1e4 # .
#SIGMA_VZ = 0.9e4 # .
#VD = np.deg2rad(36) # .
VA = 0
R_REFF = 696340000.
SPEED_RES = 100
ANGLE_RES = 90
DR = 0.1

if stellar_type == "M":
    SIGMA_VX = 3.1e4
    SIGMA_VY = 2.3e4
    SIGMA_VZ = 1.6e4
    VD = np.deg2rad(7)

elif stellar_type == "G":
    SIGMA_VX = 2.6e4
    SIGMA_VY = 1.8e4
    SIGMA_VZ = 1.5e4
    VD = np.deg2rad(12)

elif stellar_type == "OB":
    SIGMA_VX = 1.2e4
    SIGMA_VY = 1.1e4
    SIGMA_VZ = 0.9e4
    VD = np.deg2rad(36)
     
# brojaci
total_capture_count = 0
total_isos_integrated = 0
batch_count = 0

# base sim
epoha = "JD2456200.5000"
base_sim = rebound.Simulation()
base_sim.integrator = "ias15"

base_sim.add("Sun", date=epoha)
base_sim.particles[0].m = base_sim.particles[0].m + 1.6601141530543488e-07 
base_sim.add("venus", date=epoha)
base_sim.add("earth", date=epoha)
base_sim.add("mars", date=epoha)
base_sim.add("jupiter", date=epoha)
base_sim.add("saturn", date=epoha)
base_sim.add("uran", date=epoha)
base_sim.add("neptun", date=epoha)

base_sim.move_to_com()
base_sim.N_active = 8

year = 2. * np.pi
t_max = TIME_PER_BATCH_YEARS * year
dt = STEP_YEARS * year
times = np.arange(0., t_max, dt)
total_steps_in_batch = len(times)

with open(CAPTURE_LOG_FILE, 'a') as capture_f, open(PROGRESS_LOG_FILE, 'a') as progress_f:
    # glavna petlja
    while True:
        batch_count += 1
        print(f"\n--- Starting Batch {batch_count} ---")

        epoha = f"JD{epoha_0 + random.uniform(0, max_dana):.4f}" # pocetna epoha za ovu integraciju
        sim = rebound.Simulation()
        sim.integrator = "ias15"
        
        sim.add("Sun", date=epoha)
        sim.particles[0].m = base_sim.particles[0].m + 1.6601141530543488e-07 
        sim.add("venus", date=epoha)
        sim.add("earth", date=epoha)
        sim.add("mars", date=epoha)
        sim.add("jupiter", date=epoha)
        sim.add("saturn", date=epoha)
        sim.add("uran", date=epoha)
        sim.add("neptun", date=epoha)
        
        sim.move_to_com()
        sim.N_active = 8

        print("Generating new synthetic population...")
        q, e, f, inc, node, argument, _, _, _ = synthetic_population_shell(
                rmin=r_min, rmax =r_max,  n0=N0, v_min=1e1, v_max=2.5e4, 
                u_Sun=U_SUN, v_Sun=V_SUN, w_Sun=W_SUN, 
                sigma_vx=SIGMA_VX, sigma_vy=SIGMA_VY, sigma_vz=SIGMA_VZ, 
                vd=VD, va=VA, R_reff=R_REFF,
                speed_resolution=SPEED_RES, angle_resolution=ANGLE_RES, dr=DR,
                d_ref=1000, d=[], alpha=[]
            )
        
        a = q / (1 - e)
        r = (a*(1-e**2))/(1+e*np.cos(f))
        selection = np.logical_and(q<30, f<0)
        q = q[selection]
        e = e[selection]
        f = f[selection]
        inc = inc[selection]
        node = node[selection]
        argument = argument[selection]
        a = q / (1 - e)

        print(f"Adding {len(a)} new ISOs to simulation...")

        for i in range(len(a)):
                sim.add(a=a[i], e=e[i], inc=inc[i], Omega=node[i], omega=argument[i], f=f[i])

        num_isos_in_batch = len(a)
            
        total_isos_integrated += num_isos_in_batch
        
        progress_line = f"{time.ctime()}: Batch {batch_count} - {total_capture_count}/{total_isos_integrated} je uhvaceno\n"
        progress_f.write(progress_line)
        progress_f.flush()

        active_objects = list(range(len(a)))

        chunk_years = 500
        chunk_time = chunk_years * 2. * np.pi
        current_t = 0.0

        while current_t < t_max:
            current_t += chunk_time
            sim.integrate(current_t)
            
            # idemo unazad kroz petlju jer brišemo čestice
            # ova granica sigurno može da bude i manja
            for j in range(sim.N - 1, sim.N_active - 1, -1):
                p = sim.particles[j]
                r_sq = p.x**2 + p.y**2 + p.z**2
                
                # proveravamo je l asteroid daleko a nije uhvaćen
                if r_sq > 1000**2:
                    try:
                        if p.orbit().e >= 1.0:
                            sim.remove(index=j)
                            del active_objects[j - sim.N_active]
                    except rebound.ParticleNotFound:
                        pass
                        
            # završavamo batch ako su svi objekti izbačeni iz simulacije
            if sim.N == sim.N_active:
                break

        for j in range(sim.N_active, sim.N):
            iso_index_in_batch = active_objects[j - sim.N_active]
            
            try:
                orb_b = sim.particles[j].orbit()

            except rebound.ParticleNotFound:
                continue
            
            if orb_b.e < 1.0: 
                log_line = ( 
                    f"CAPTURE: [BARYCENTRIC] Batch: {batch_count}, ISO Index: {iso_index_in_batch}, " 
                    f"Epoch: {epoha}, "  # <--- DODATO OVDE
                    f"a: {orb_b.a}, e: {orb_b.e}, i: {np.rad2deg(orb_b.inc)}, " 
                    f"Omega: {np.rad2deg(orb_b.Omega)}, omega: {np.rad2deg(orb_b.omega)}\n" 
            
                    f"CAPTURE: [INITIAL] Batch: {batch_count}, ISO Index: {iso_index_in_batch}, " 
                    f"Epoch: {epoha}, "  # <--- DODATO OVDE
                    f"a: {a[iso_index_in_batch]}, e: {e[iso_index_in_batch]}, i: {np.rad2deg(inc[iso_index_in_batch])}, " 
                    f"Omega: {np.rad2deg(node[iso_index_in_batch])}, omega: {np.rad2deg(argument[iso_index_in_batch])}, f: {np.rad2deg(f[iso_index_in_batch])}\n" 
            
                    f"-------------------------------------------------------------------------------------\n" 
                ) 
                capture_f.write(log_line) 
                capture_f.flush() 
                
                total_capture_count += 1
