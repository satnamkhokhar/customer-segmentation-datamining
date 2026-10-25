"""Synthetic fixtures are used only for correctness checks, never research results."""
import numpy as np
import pandas as pd
import pytest
from src.data_loader import validate_columns, load_data
from src.preprocessing import clean_transactions
from src.rfm_analysis import calculate_rfm
from src.clustering import prepare_features, run_kmeans, profile_segments
from threadpoolctl import threadpool_limits


@pytest.fixture
def purchases():
    return pd.DataFrame([
        ['1', 'A', 'Item', 2, '2011-12-01 12:00', 3., 10., 'UK'],
        ['1', 'B', 'Other', 1, '2011-12-01 12:00', 5., 10., 'UK'],
        ['2', 'A', 'Item', 1, '2011-12-03 10:00', 4., 10., 'UK'],
        ['3', 'A', 'Item', 2, '2011-12-04 16:00', 10., 20., 'UK'],
    ], columns=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country'])


def test_required_columns(purchases):
    with pytest.raises(ValueError, match='Missing required columns'):
        validate_columns(purchases.drop(columns='Quantity'))


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match='--download'):
        load_data(tmp_path / 'missing.xlsx')


@pytest.mark.parametrize('field,value,step', [
    ('CustomerID',np.nan,'missing_customer_id'), ('InvoiceNo','C999','cancelled_invoices'),
    ('Quantity',0,'invalid_quantity'), ('Quantity',-1,'invalid_quantity'),
    ('UnitPrice',0.,'invalid_price'), ('UnitPrice',-1.,'invalid_price'),
    ('InvoiceDate','invalid','invalid_date'), ('Quantity',np.inf,'invalid_quantity'),
    ('UnitPrice',np.nan,'invalid_price'), ('InvoiceNo',None,'missing_invoice')])
def test_cleaning_rules(purchases, field, value, step):
    purchases[field] = purchases[field].astype(object)
    purchases.loc[0, field] = value
    clean, ledger = clean_transactions(purchases)
    assert len(clean) == 3
    assert ledger.set_index('step').loc[step,'rows_removed'] == 1
    assert ledger.rows_removed.sum() == len(purchases) - len(clean)


def test_duplicates_and_legitimate_invoice_lines(purchases):
    source = pd.concat([purchases, purchases.iloc[[0]]], ignore_index=True)
    clean, ledger = clean_transactions(source)
    assert len(clean) == 4
    assert (clean.InvoiceNo == '1').sum() == 2
    assert ledger.set_index('step').loc['exact_duplicates','rows_removed'] == 1
    assert clean.TotalPrice.tolist() == [6.,5.,4.,20.]
    assert clean.CustomerID.tolist() == ['10','10','10','20']


def test_rfm_exact_calculations(purchases):
    clean, _ = clean_transactions(purchases)
    rfm, date = calculate_rfm(clean)
    assert date == pd.Timestamp('2011-12-05')
    assert rfm.CustomerID.is_unique
    assert rfm.set_index('CustomerID').loc['10'].to_dict() == {'Recency':2,'Frequency':2,'Monetary':15.}
    assert rfm.set_index('CustomerID').loc['20'].to_dict() == {'Recency':1,'Frequency':1,'Monetary':20.}


def test_cluster_assignments_and_scaling():
    rfm = pd.DataFrame({'CustomerID': [str(i) for i in range(18)],
                        'Recency': np.arange(18)+1, 'Frequency': np.arange(18)+1,
                        'Monetary': (np.arange(18)+1)**2})
    features, _, transformed = prepare_features(rfm)
    np.testing.assert_allclose(features.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(transformed.Frequency, np.log1p(rfm.Frequency))
    with threadpool_limits(limits=2):
        comparison, models, assignments = run_kmeans(features)
    assert comparison.k.tolist() == list(range(2,11))
    for k, labels in assignments.items():
        assert len(labels) == len(rfm)
        assert set(labels) == set(range(k))
        np.testing.assert_array_equal(models[k].predict(features), labels)
    assigned, profile = profile_segments(rfm, assignments[3])
    assert profile.customers.sum() == len(rfm)
    assert assigned.CustomerID.is_unique


def test_degenerate_customers():
    with pytest.raises(ValueError, match='three customers'):
        run_kmeans(np.ones((4,3)))


def test_end_to_end_exports_on_fixture(tmp_path):
    """Exercise exports on a synthetic workbook confined to pytest's temp directory."""
    import json
    from main import run_pipeline
    rows = []
    for customer in range(24):
        for purchase in range(1 + customer % 5):
            rows.append([f'{customer}-{purchase}', 'A', 'Test fixture', 1 + customer,
                         pd.Timestamp('2011-12-01') - pd.Timedelta(days=customer*3-purchase),
                         1.5 + customer % 3, customer + 100, 'UK'])
    workbook = tmp_path / 'fixture.xlsx'
    pd.DataFrame(rows, columns=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']).to_excel(workbook, index=False)
    summary = run_pipeline(workbook, output_root=tmp_path)
    assert summary['dataset']['customers'] == 24
    assert summary['kmeans']['tested_k'] == list(range(2,11))
    assert len(list((tmp_path / 'results/figures').glob('*.png'))) == 11
    exported = json.loads((tmp_path / 'results/tables/experimental_summary.json').read_text())
    assert exported['dataset']['original_transactions'] == len(rows)
    assert (tmp_path / 'results/models/selected_kmeans.pkl').exists()
    assignments = pd.read_csv(tmp_path / 'data/processed/customer_segments.csv')
    assert len(assignments) == 24
    assert assignments.Cluster.nunique() == summary['kmeans']['selected_k']
