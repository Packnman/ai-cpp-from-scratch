# TokenCharacter 設計仕様

対象ヘッダ: `lib/include/tokenizer_character.h`。

## 責務・公開 API

`TokenCharacter( std::string_view corpus )` は UTF-8 を Unicode コードポイントに分解し、昇順・重複なしの語彙を構築する。正規化、形態素解析、byte 分割は行わない。

既存の `encode(text)` は未知文字を `out_of_range` とする。追加 API `encodeUnknown( std::string_view c_strText, std::int32_t nUnknownId )` は未知コードポイントごとに指定 ID を出す。既存 API の標準動作を変更しない。`decode(TokenIds)` は通常 ID を UTF-8 に戻し、`vocabSize()` は通常語彙数を返す。既定の特殊 ID は存在しない。

## データ・所有権・処理

入力文字列は借用し、構築時にコードポイント配列と逆引き表を値として所有する。encode/decode の返値は呼出し側が所有する。GPU Tensor は作らない。処理は UTF-8 検証→コードポイント lookup→ID 列。全て CPU の非微分処理であり、数式・勾配、学習／評価のモード差はない。語彙は構築後に変更しない。

## 例外・保存・完了条件

不正 UTF-8（不正継続 byte、overlong、surrogate、範囲外、切断）は `invalid_argument`。decode の不正 ID と strict encode の未知文字は `out_of_range`。語彙上限は int32。単体に保存 API はなく、会話 wrapper が昇順の全文字列を bundle v1 に保存し、再構築時も同じ ID になる。

既存 `character_tokenizer_check` と `conversation_check` の Unicode・未知文字・不正 UTF-8・strict 動作維持を完了条件とする。
