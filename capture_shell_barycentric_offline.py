import numpy as np
import rebound
import time
import random
import sys
import spiceypy as spice

# generator
from synthetic_population_shell import synthetic_population_shell

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
N0 = 1e-1
V_MIN = 1e3
V_MAX = 2e5
U_SUN = 1e4
V_SUN = 1.1e4
W_SUN = 7e3
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

# spice setup, skinuti ove fajlove pre pokretanja
spice.kclear()
spice.furnsh("naif0012.tls")
spice.furnsh("de441_part-2.bsp")
spice.furnsh("gm_de440.tpc")

AU_KM = 149597870.7  
k_gauss = 0.01720209895
REBOUND_TIME_S = 86400.0 / k_gauss
# konverzija iz SPICE GM [km^3/s^2] u rebound jedinice mase
GM_TO_REBOUND_MASS = (REBOUND_TIME_S**2) / (AU_KM**3)

bodies_map = [
    ("Sun", 10), ("venus", 2), ("earth", 3), ("mars", 4),
    ("jupiter", 5), ("saturn", 6), ("uran", 7), ("neptun", 8)
]

# iz spice fajlova racunamo mase planeta u suncevim masama koristeci gausovu konstantu k
precalc_masses = {}
for name, naif_id in bodies_map:
    raw_gm = spice.bodvcd(naif_id, "GM", 1)[1][0]
    precalc_masses[name] = raw_gm * GM_TO_REBOUND_MASS

# dodajemo masu merkura na sunce - otkomentarisati naredne dve linije i zakomentarisati trecu ako zelimo precizan rezultat
# raw_mercury_gm = spice.bodvcd(1, "GM", 1)[1][0]
# precalc_masses["Sun"] += raw_mercury_gm * GM_TO_REBOUND_MASS
precalc_masses["Sun"] += 1.6601141530543488e-07

# brojaci
total_capture_count = 0
total_isos_integrated = 0
batch_count = 0

year = 2. * np.pi
t_max = TIME_PER_BATCH_YEARS * year

with open(CAPTURE_LOG_FILE, 'a') as capture_f, open(PROGRESS_LOG_FILE, 'a') as progress_f:
    # glavna petlja
    while True:
        batch_count += 1
        print(f"\n--- Starting Batch {batch_count} ---")

        epoha_float = epoha_0 + random.uniform(0, max_dana)
        epoha_str = f"JD{epoha_float:.4f}"
        
        sim = rebound.Simulation()
        sim.integrator = "ias15"
        
        # konverzija iz JD u Ephemeris Time koji SPICE koristi
        et = spice.unitim(epoha_float, "JDTDB", "ET") 
    
        # offline dodavanje planeta kroz SPICE
        for name, naif_id in bodies_map:
            mass = precalc_masses[name]
            
            # postavljamo sunce (id 10) u centar i planete oko njega, pa kasnije pomeramo baricentar u 0 0 0
            if naif_id == 10:
                sim.add(m=mass, x=0, y=0, z=0, vx=0, vy=0, vz=0, hash=name)
                continue
            
            # spice koristi ekvatorske koordinate, pa tražimo ekliptičke (koje rebound i mi koristimo)
            state, _ = spice.spkezr(str(naif_id), et, "ECLIPJ2000", "NONE", "10")
            
            x, y, z = [pos / AU_KM for pos in state[:3]]
            # skaliranje brzine u rebound jedinice
            vx, vy, vz = [(vel / AU_KM) * REBOUND_TIME_S for vel in state[3:]]
            
            sim.add(m=mass, x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, hash=name)
            
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
            try:
                sim.integrate(current_t)

            except rebound.IntegrationError:
                error_msg = f"{time.ctime()}: [WARNING] Integration Error (collision) in Batch {batch_count}. Skipping.\n"
                progress_f.write(error_msg)
                progress_f.flush()
                break  

            # idemo unazad kroz petlju jer brišemo čestice
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
                    f"Epoch: {epoha_str}, " 
                    f"a: {orb_b.a}, e: {orb_b.e}, i: {np.rad2deg(orb_b.inc)}, " 
                    f"Omega: {np.rad2deg(orb_b.Omega)}, omega: {np.rad2deg(orb_b.omega)}\n" 
            
                    f"CAPTURE: [INITIAL] Batch: {batch_count}, ISO Index: {iso_index_in_batch}, " 
                    f"Epoch: {epoha_str}, " 
                    f"a: {a[iso_index_in_batch]}, e: {e[iso_index_in_batch]}, i: {np.rad2deg(inc[iso_index_in_batch])}, " 
                    f"Omega: {np.rad2deg(node[iso_index_in_batch])}, omega: {np.rad2deg(argument[iso_index_in_batch])}, f: {np.rad2deg(f[iso_index_in_batch])}\n" 
            
                    f"-------------------------------------------------------------------------------------\n" 
                ) 
                capture_f.write(log_line) 
                capture_f.flush() 
                
                total_capture_count += 1