from frontend.streamlit_app.services.upload_parser import canonical_upload_field, normalize_header_name


def test_normalize_header_name_collapses_symbols():
    assert normalize_header_name("月经周期（天）") == "月经周期天"


def test_canonical_upload_field_matches_common_aliases():
    assert canonical_upload_field("患者姓名") == "patient_name"
    assert canonical_upload_field("基础FSH") == "FSH"
