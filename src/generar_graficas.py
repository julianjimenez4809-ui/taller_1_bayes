import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import math

# Parámetros Globales
y_observed = 55
x = np.linspace(0, 150, 1000)

# --- ESCENARIO A: PRIOR INFORMADA (COMERCIANTES) ---
alpha_prior_A = 12.5
beta_prior_A = 0.25

# Posterior A
alpha_post_A = alpha_prior_A + y_observed
beta_post_A = beta_prior_A + 1
media_post_A = alpha_post_A / beta_post_A
low_95_post_A = gamma.ppf(0.025, a=alpha_post_A, scale=1/beta_post_A)
upp_95_post_A = gamma.ppf(0.975, a=alpha_post_A, scale=1/beta_post_A)

# --- ESCENARIO B: PRIOR DÉBIL (VALENTINA) ---
alpha_prior_B = 8.0
beta_prior_B = 0.1

# Posterior B
alpha_post_B = alpha_prior_B + y_observed
beta_post_B = beta_prior_B + 1
media_post_B = alpha_post_B / beta_post_B
low_95_post_B = gamma.ppf(0.025, a=alpha_post_B, scale=1/beta_post_B)
upp_95_post_B = gamma.ppf(0.975, a=alpha_post_B, scale=1/beta_post_B)

# --- FUNCIÓN AUXILIAR PARA SOMBREAR ---
def shade_ic(x_range, dist, params, color, label_ic="IC 95%", alpha=0.3):
    low = gamma.ppf(0.025, a=params[0], scale=1/params[1])
    high = gamma.ppf(0.975, a=params[0], scale=1/params[1])
    x_fill = np.linspace(low, high, 100)
    y_fill = gamma.pdf(x_fill, a=params[0], scale=1/params[1])
    plt.fill_between(x_fill, y_fill, color=color, alpha=alpha, label=label_ic)

# --- GENERACIÓN DE GRÁFICAS ---

# G1_Prior.png (Escenario A)
plt.figure(figsize=(10, 6))
y_pA = gamma.pdf(x, a=alpha_prior_A, scale=1/beta_prior_A)
plt.plot(x, y_pA, color='teal', lw=2, label='Prior Informada')
shade_ic(x, gamma, (alpha_prior_A, beta_prior_A), 'teal')
plt.axvline(alpha_prior_A/beta_prior_A, color='red', linestyle='--', label='Media')
plt.title('G1: Prior Informada (Historica)')
plt.xlabel('$\lambda$')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G1_Prior.png')
plt.close()

# G1_Weak_Prior.png (Escenario B)
plt.figure(figsize=(10, 6))
y_pB = gamma.pdf(x, a=alpha_prior_B, scale=1/beta_prior_B)
plt.plot(x, y_pB, color='purple', lw=2, label='Prior Débil')
shade_ic(x, gamma, (alpha_prior_B, beta_prior_B), 'purple')
plt.axvline(alpha_prior_B/beta_prior_B, color='red', linestyle='--', label='Media')
plt.title('G1-Weak: Prior Débil (Valentina)')
plt.xlabel('$\lambda$')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G1_Weak_Prior.png')
plt.close()

# --- NUEVA: G2_Likelihood.png ---
plt.figure(figsize=(10, 6))
# Likelihood Poisson p(y=55 | lambda) = (e^-L * L^y) / y!
# Escalamos para que el máximo sea 1 para visualización
likelihood = gamma.pdf(x, a=y_observed + 1, scale=1) # Forma visual de la Poisson como continua
likelihood /= likelihood.max()
plt.plot(x, likelihood, color='orange', lw=2, label=f'Likelihood Poisson (y={y_observed})')
plt.axvline(y_observed, color='black', linestyle=':', label='MLE (y=55)')
plt.title('G2: Likelihood Poisson (Escalada)')
plt.xlabel('$\lambda$')
plt.ylabel('Verosimilitud Relativa')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G2_Likelihood.png')
plt.close()

# G3_Posterior.png (Escenario A)
# ... (resto del código igual)
plt.figure(figsize=(10, 6))
plt.plot(x, y_pA, '--', color='teal', alpha=0.5, label='Prior Informada')
plt.plot(x, gamma.pdf(x, a=alpha_post_A, scale=1/beta_post_A), color='darkblue', lw=2, label='Posterior A')
shade_ic(x, gamma, (alpha_post_A, beta_post_A), 'darkblue', label_ic="IC 95% Posterior")
plt.axvline(media_post_A, color='red', linestyle='-', label=f'Media Post = {media_post_A:.1f}')
plt.title('G3: Prior Informada vs Posterior')
plt.xlabel('$\lambda$')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G3_Posterior.png')
plt.close()

# G3_Weak_Posterior.png (Escenario B)
plt.figure(figsize=(10, 6))
plt.plot(x, y_pB, '--', color='purple', alpha=0.5, label='Prior Débil')
plt.plot(x, gamma.pdf(x, a=alpha_post_B, scale=1/beta_post_B), color='indigo', lw=2, label='Posterior B')
shade_ic(x, gamma, (alpha_post_B, beta_post_B), 'indigo', label_ic="IC 95% Posterior")
plt.axvline(media_post_B, color='red', linestyle='-', label=f'Media Post = {media_post_B:.1f}')
plt.title('G3-Weak: Prior Débil vs Posterior')
plt.xlabel('$\lambda$')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G3_Weak_Posterior.png')
plt.close()

from scipy.stats import gamma, nbinom

# ... (parámetros anteriores se mantienen)

# --- ESCENARIO A: PREDICTIVA ---
# La predictiva Gamma-Poisson es Negative Binomial con n=alpha_post y p=beta_post/(beta_post+1)
n_A = alpha_post_A
p_A = beta_post_A / (beta_post_A + 1)
# Media predictiva = E[Y|y] = alpha/beta (coincide con media posterior)
media_pred_A = media_post_A
upper_95_pred_A = nbinom.ppf(0.95, n_A, p_A)

# --- ESCENARIO B: PREDICTIVA ---
n_B = alpha_post_B
p_B = beta_post_B / (beta_post_B + 1)
media_pred_B = media_post_B
upper_95_pred_B = nbinom.ppf(0.95, n_B, p_B)

# --- GENERACIÓN DE G4 ---
def plot_predictive(n, p, title, filename, color):
    # Valores posibles de turistas (0 a 120 por ejemplo)
    y_vals = np.arange(0, 120)
    pmf_vals = nbinom.pmf(y_vals, n, p)
    
    plt.figure(figsize=(10, 6))
    plt.bar(y_vals, pmf_vals, color=color, alpha=0.4, label='Predictiva Posterior (NB)')
    
    # Intervalo 95%
    low = int(nbinom.ppf(0.025, n, p))
    high = int(nbinom.ppf(0.975, n, p))
    y_fill = np.arange(low, high + 1)
    plt.bar(y_fill, nbinom.pmf(y_fill, n, p), color=color, alpha=0.8, label='IP 95%')
    
    plt.axvline(n/((1-p)/p), color='red', linestyle='--', label=f'Media = {n/((1-p)/p):.1f}')
    plt.axvline(nbinom.ppf(0.95, n, p), color='black', linestyle=':', label=f'P95 (Prudente) = {nbinom.ppf(0.95, n, p):.0f}')
    
    plt.title(title)
    plt.xlabel('Número de turistas (Y_mañana)')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(filename)
    plt.close()

plot_predictive(n_A, p_A, 'G4: Predictiva Posterior (Prior Informada)', 'graficas/G4_Predictiva.png', 'teal')
plot_predictive(n_B, p_B, 'G4: Predictiva Posterior (Prior Débil)', 'graficas/G4_Weak_Predictiva.png', 'purple')

print(f"Escenario A (Informada): Media={media_pred_A:.1f}, Prudente (P95)={upper_95_pred_A:.0f}")
print(f"Escenario B (Débil): Media={media_pred_B:.1f}, Prudente (P95)={upper_95_pred_B:.0f}")
print("Todas las gráficas generadas (G1, G2, G3, G4).")

# --- PARTE B: BETA-BINOMIAL ---
from scipy.stats import beta, binom, betabinom

# B1. Prior Local
alpha_prior_beta = 4.4
beta_prior_beta = 6.6

p_grid = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
plt.plot(p_grid, beta.pdf(p_grid, alpha_prior_beta, beta_prior_beta), color='green', lw=2, label='Prior Beta(4.4, 6.6)')
# Sombreado IC 95%
low_b = beta.ppf(0.025, alpha_prior_beta, beta_prior_beta)
high_b = beta.ppf(0.975, alpha_prior_beta, beta_prior_beta)
p_fill = np.linspace(low_b, high_b, 100)
plt.fill_between(p_fill, beta.pdf(p_fill, alpha_prior_beta, beta_prior_beta), color='green', alpha=0.3, label='IC 95%')
plt.axvline(alpha_prior_beta/(alpha_prior_beta+beta_prior_beta), color='red', linestyle='--', label=f'Media = {alpha_prior_beta/(alpha_prior_beta+beta_prior_beta):.2f}')

plt.title('G1_Beta: Prior Local de proporción de turistas')
plt.xlabel('Proporción (p)')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G1_Beta_Prior.png')
plt.close()

# B2. Dato Observado: x=42, n=100
x_obs = 42
n_obs = 100

# Likelihood Binomial: p(x=42 | p) = p^42 * (1-p)^58 (ignorando combinatoria)
likelihood_beta = (p_grid**x_obs) * ((1 - p_grid)**(n_obs - x_obs))
likelihood_beta /= likelihood_beta.max() # Escalamos a 1 para visualización

plt.figure(figsize=(10, 6))
plt.plot(p_grid, likelihood_beta, color='orange', lw=2, label=f'Likelihood Binomial (x={x_obs}, n={n_obs})')
plt.axvline(x_obs/n_obs, color='black', linestyle=':', label=f'MLE (x/n = {x_obs/n_obs:.2f})')
plt.title('G2_Beta: Likelihood Binomial (Escalada)')
plt.xlabel('Proporción (p)')
plt.ylabel('Verosimilitud Relativa')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G2_Beta_Likelihood.png')
plt.close()

# B2.3. Posterior conjugada
alpha_post_beta = alpha_prior_beta + x_obs
beta_post_beta = beta_prior_beta + (n_obs - x_obs)

plt.figure(figsize=(10, 6))
plt.plot(p_grid, beta.pdf(p_grid, alpha_prior_beta, beta_prior_beta), '--', color='green', alpha=0.5, label='Prior Local')
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_beta, beta_post_beta), color='darkgreen', lw=2, label=f'Posterior Beta({alpha_post_beta}, {beta_post_beta})')

low_post = beta.ppf(0.025, alpha_post_beta, beta_post_beta)
high_post = beta.ppf(0.975, alpha_post_beta, beta_post_beta)
p_fill_post = np.linspace(low_post, high_post, 100)
plt.fill_between(p_fill_post, beta.pdf(p_fill_post, alpha_post_beta, beta_post_beta), color='darkgreen', alpha=0.3, label='IC 95% Posterior')
plt.axvline(alpha_post_beta/(alpha_post_beta+beta_post_beta), color='red', linestyle='--', label=f'Media Post = {alpha_post_beta/(alpha_post_beta+beta_post_beta):.2f}')

plt.title('G3_Beta: Prior Local vs Posterior')
plt.xlabel('Proporción (p)')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G3_Beta_Posterior.png')
plt.close()

print(f"Beta Posterior Media: {alpha_post_beta/(alpha_post_beta+beta_post_beta):.3f}")

# --- B3. PRIOR NO INFORMATIVA (NICOLAS) ---
alpha_prior_nic = 1
beta_prior_nic = 1

alpha_post_nic = alpha_prior_nic + x_obs
beta_post_nic = beta_prior_nic + (n_obs - x_obs)

plt.figure(figsize=(10, 6))
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_beta, beta_post_beta), color='darkgreen', lw=2, label='Posterior B2 (Experiencia Local)')
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_nic, beta_post_nic), color='navy', lw=2, linestyle='--', label='Posterior B3 (Ignorancia Total)')

# Sombreado IC 95% Posterior Local (B2)
low_b2 = beta.ppf(0.025, alpha_post_beta, beta_post_beta)
high_b2 = beta.ppf(0.975, alpha_post_beta, beta_post_beta)
p_fill_b2 = np.linspace(low_b2, high_b2, 100)
plt.fill_between(p_fill_b2, beta.pdf(p_fill_b2, alpha_post_beta, beta_post_beta), color='darkgreen', alpha=0.3, label='IC 95% Exp. Local')

# Sombreado IC 95% Posterior Nicolás (B3)
low_b3 = beta.ppf(0.025, alpha_post_nic, beta_post_nic)
high_b3 = beta.ppf(0.975, alpha_post_nic, beta_post_nic)
p_fill_b3 = np.linspace(low_b3, high_b3, 100)
plt.fill_between(p_fill_b3, beta.pdf(p_fill_b3, alpha_post_nic, beta_post_nic), color='navy', alpha=0.3, label='IC 95% Ignorancia Total')

plt.title('G3_Comparativa: Posteriores con Prior Local vs Prior Uniforme')
plt.xlabel('Proporción (p)')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G3_Comparacion_Posteriores.png')
plt.close()

# --- B4. EXPERTO TERCO ---
alpha_prior_terco = 320
beta_prior_terco = 80

alpha_post_terco = alpha_prior_terco + x_obs
beta_post_terco = beta_prior_terco + (n_obs - x_obs)
media_post_terco = alpha_post_terco / (alpha_post_terco + beta_post_terco)

plt.figure(figsize=(10, 6))

# 1. Posterior B2 (Experiencia Local) - Verde
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_beta, beta_post_beta), color='darkgreen', lw=2, label='Posterior Local')
p_fill_b2 = np.linspace(beta.ppf(0.025, alpha_post_beta, beta_post_beta), beta.ppf(0.975, alpha_post_beta, beta_post_beta), 100)
plt.fill_between(p_fill_b2, beta.pdf(p_fill_b2, alpha_post_beta, beta_post_beta), color='darkgreen', alpha=0.3, label='IC 95% Local')
plt.axvline(alpha_post_beta/(alpha_post_beta+beta_post_beta), color='darkgreen', linestyle=':', alpha=0.7)

# 2. Posterior B3 (Ignorancia Total) - Azul
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_nic, beta_post_nic), color='navy', lw=2, linestyle='--', label='Posterior Uniforme')
p_fill_b3 = np.linspace(beta.ppf(0.025, alpha_post_nic, beta_post_nic), beta.ppf(0.975, alpha_post_nic, beta_post_nic), 100)
plt.fill_between(p_fill_b3, beta.pdf(p_fill_b3, alpha_post_nic, beta_post_nic), color='navy', alpha=0.3, label='IC 95% Uniforme')
plt.axvline(alpha_post_nic/(alpha_post_nic+beta_post_nic), color='navy', linestyle=':', alpha=0.7)

# 3. Posterior B4 (Experto Terco) - Rojo
plt.plot(p_grid, beta.pdf(p_grid, alpha_post_terco, beta_post_terco), color='firebrick', lw=2, label='Posterior Experto Terco')
p_fill_terco = np.linspace(beta.ppf(0.025, alpha_post_terco, beta_post_terco), beta.ppf(0.975, alpha_post_terco, beta_post_terco), 100)
plt.fill_between(p_fill_terco, beta.pdf(p_fill_terco, alpha_post_terco, beta_post_terco), color='firebrick', alpha=0.3, label='IC 95% Terco')
plt.axvline(media_post_terco, color='firebrick', linestyle=':', alpha=0.7)


plt.title('G3_Comparativa_Tres: Posteriores (Local vs Uniforme vs Experto Terco)')
plt.xlabel('Proporción (p)')
plt.ylabel('Densidad')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('graficas/G3_Comparacion_Tres_Posteriores.png')
plt.close()

print(f"Media Posterior Terco: {media_post_terco:.3f}")

# --- B5. PREDICCIONES BETA-BINOMIAL (m = 50) ---
m_new = 50
x_new_vals = np.arange(0, m_new + 1)

def graficar_predictiva_beta_binom(alpha_p, beta_p, m, nombre_escenario, filename, color):
    # betabinom.pmf(k, n, a, b) en scipy
    pmf_vals = betabinom.pmf(x_new_vals, m, alpha_p, beta_p)
    
    plt.figure(figsize=(10, 6))
    plt.bar(x_new_vals, pmf_vals, color=color, alpha=0.4, label=f'Predictiva {nombre_escenario}')
    
    # Intervalo 95%
    low_pred = int(betabinom.ppf(0.025, m, alpha_p, beta_p))
    high_pred = int(betabinom.ppf(0.975, m, alpha_p, beta_p))
    
    x_fill = np.arange(low_pred, high_pred + 1)
    plt.bar(x_fill, betabinom.pmf(x_fill, m, alpha_p, beta_p), color=color, alpha=0.8, label=f'IP 95% [{low_pred}, {high_pred}]')
    
    media_pred = m * (alpha_p / (alpha_p + beta_p))
    plt.axvline(media_pred, color='red', linestyle='--', label=f'Media = {media_pred:.1f}')
    
    plt.title(f'G4_Beta: Predictiva Posterior ({nombre_escenario})')
    plt.xlabel('Número de turistas que se quedarán (x_mañana)')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename)
    plt.close()
    
    print(f"Predictiva {nombre_escenario} -> Media: {media_pred:.1f}, IP 95%: [{low_pred}, {high_pred}]")

graficar_predictiva_beta_binom(alpha_post_beta, beta_post_beta, m_new, "Local", 'graficas/G4_Beta_Predictiva_Local.png', 'darkgreen')
graficar_predictiva_beta_binom(alpha_post_nic, beta_post_nic, m_new, "Uniforme", 'graficas/G4_Beta_Predictiva_Uniforme.png', 'navy')
graficar_predictiva_beta_binom(alpha_post_terco, beta_post_terco, m_new, "Experto Terco", 'graficas/G4_Beta_Predictiva_Terco.png', 'firebrick')
