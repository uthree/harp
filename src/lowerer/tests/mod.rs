// グラフ最適化が必須になったため、lowerer固有のテストは削除
// LoweringSuggesterのテストは opt/graph/suggesters/lowering.rs にあります
mod signature_tests; // シグネチャ生成テスト（Lowerer::create_signature）
mod topological_tests; // トポロジカルソート＆コード生成テスト
