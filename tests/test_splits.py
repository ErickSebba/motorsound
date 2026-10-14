import numpy as np
import pytest

from motorsound import io, splits


@pytest.fixture(scope="module")
def segs_idmt():
    return io.segmentos(io.gravacoes("idmt", "train"), 3.0)


@pytest.fixture(scope="module")
def segs_a2212():
    return io.segmentos(io.gravacoes("a2212"), 1.0)


def test_teste_trancado(monkeypatch):
    monkeypatch.delenv("MOTORSOUND_ABRIR_TESTE", raising=False)
    with pytest.raises(io.TesteTrancado):
        io.gravacoes("idmt", "test")
    with pytest.raises(io.TesteTrancado):
        io.carregar(str(io.IDMT / "test" / "engine1_good" / "atmo_low.wav"), 16000)


def test_idmt_segmentos_batem_com_train_cut(segs_idmt):
    # 105 / 124 / 128 recortes de 3 s, como no train_cut/ (docs/DATASETS.md)
    from collections import Counter
    assert Counter(s.classe for s in segs_idmt) == {"good": 105, "broken": 124,
                                                    "heavyload": 128}


def test_idmt_blocos_disjuntos(segs_idmt):
    folds = splits.idmt_blocos(segs_idmt)
    bloco = splits.grupos_idmt_blocos(segs_idmt)
    g = splits.grupo_de(segs_idmt, bloco)
    vistos = set()
    for f in folds:
        assert not set(g[f.treino]) & set(g[f.val])
        # cada fold tem as três classes dos dois lados
        for lado in (f.treino, f.val):
            assert {segs_idmt[i].classe for i in lado} == set(io.CLASSES_IDMT)
        # nenhum segmento de treino é vizinho imediato de um de validação
        val_pos = {(segs_idmt[i].rec_id, segs_idmt[i].idx) for i in f.val}
        for i in f.treino:
            s = segs_idmt[i]
            assert (s.rec_id, s.idx - 1) not in val_pos and (s.rec_id, s.idx + 1) not in val_pos
        vistos |= set(f.val)
    # todo segmento não-guarda é validado exatamente uma vez
    assert vistos == set(np.flatnonzero(bloco >= 0))


def test_a2212_por_arquivo_disjunto(segs_a2212):
    folds = splits.a2212_por_arquivo(segs_a2212)
    rec = splits.grupo_de(segs_a2212)
    for f in folds:
        assert not set(rec[f.treino]) & set(rec[f.val])


def test_verificar_detecta_vazamento(segs_a2212):
    errado = splits.aleatorio_por_segmento(segs_a2212)
    rec = splits.grupo_de(segs_a2212)
    assert any(set(rec[f.treino]) & set(rec[f.val]) for f in errado)
    falso = [splits.Fold(f.nome, f.treino, f.val, vazamento_intencional=False) for f in errado]
    with pytest.raises(splits.VazamentoDeGrupo):
        splits.verificar(segs_a2212, falso, rec)
