// Minimal SCF/arith representation of the coin-flip dynamic circuit loop.
// Used by E3 to benchmark WhileLoopPeelingPass with --peel-factor.
//
// All loop-carried state (count, result) threads through both before/do regions.

module {
  func.func @coin_flip(%init_result: i64) -> i64 {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64

    %count_out, %result_out = scf.while (%count = %c0, %result = %init_result)
        : (i64, i64) -> (i64, i64) {
      %cond = arith.cmpi eq, %result, %c0 : i64
      scf.condition(%cond) %count, %result : i64, i64
    } do {
    ^bb0(%count : i64, %result : i64):
      %new_count = arith.addi %count, %c1 : i64
      // Simulate measure: flip result bit (in real circuit this is quantum measure).
      %flipped = arith.subi %c1, %result : i64
      scf.yield %new_count, %flipped : i64, i64
    }

    func.return %count_out : i64
  }
}
