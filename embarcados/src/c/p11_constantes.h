/* p11_constantes.h -- GERADO AUTOMATICAMENTE, não editar à mão.
 * Origem: embarcados/src/python/gerar_constantes.py (commit 9081500),
 *         scipy 1.18.1, numpy 2.5.4,
 *         gerado em 2026-10-24T02:28:39Z.
 * Conteúdo: scipy.signal.butter(4, 200, fs=16000, output="sos"), o passa-baixa de
 *           pipelines/_decifrae.py::_envelope_mod. Cada linha = [b0 b1 b2 a0 a1 a2] (a0 = 1).
 * sha256(sos float64)[:16] = 45e882fb44c75491
 */
#ifndef P11_CONSTANTES_H
#define P11_CONSTANTES_H

#define P11_SOS_N_SECOES 2

/* repr float64 (ida e volta exata) */
static const double P11_SOS_F64[P11_SOS_N_SECOES][6] = {
    {2.1505687372880113e-06, 4.301137474576023e-06, 2.1505687372880113e-06, 1.0, -1.8590762659582096, 0.8648248987672625},
    {1.0, 2.0, 1.0, 1.0, -1.9357148371211979, 0.9417004516037268},
};

/* os mesmos valores arredondados para float32 (build float) */
static const float P11_SOS_F32[P11_SOS_N_SECOES][6] = {
    {2.1505686618183972e-06f, 4.3011373236367945e-06f, 2.1505686618183972e-06f, 1.0f, -1.8590762615203857f, 0.8648248910903931f},
    {1.0f, 2.0f, 1.0f, 1.0f, -1.935714840888977f, 0.9417004585266113f},
};

#endif /* P11_CONSTANTES_H */
