module divider_tb;
  logic clk = 0;
  always #5 clk = ~clk;
  logic reset = 1, go = 0, sqrtOp = 0;
  logic [31:0] left = 0, right = 0;
  logic [2:0] roundingMode = 0;
  wire [31:0] result;
  wire [4:0] flags;
  wire done;
  logic main_go = 0;
  wire main_done;
  wire [31:0] main_a, main_b, main_c;
  main generated_control(.clk(clk), .reset(reset), .go(main_go), .done(main_done),
    .a(main_a), .b(main_b), .c(main_c));
  integer cycles = 0, accepted = 0, completed = 0, transactions = 0;
  std_divSqrtFN dut(.clk(clk), .reset(reset), .go(go), .control(1'b0),
    .sqrtOp(sqrtOp), .left(left), .right(right), .roundingMode(roundingMode),
    .out(result), .exceptionFlags(flags), .done(done));
  always @(posedge clk) begin
    cycles <= cycles + 1;
    if (cycles > 5000) $fatal(1, "timeout");
    if (!reset && dut.divSqrtRecFNToRaw.inReady && dut.divSqrtRecFNToRaw.inValid)
      accepted <= accepted + 1;
    if (!reset && dut.outValid) completed <= completed + 1;
  end

  task check(input logic sq, input logic [31:0] a, b, expected,
             input logic [4:0] expected_flags, input integer gap);
    begin
      @(negedge clk);
      sqrtOp = sq; left = a; right = b; go = 1;
      while (!done) @(negedge clk);
      transactions = transactions + 1;
      if (result !== expected || flags !== expected_flags)
        $fatal(1, "transaction %0d actual=%h flags=%h expected=%h flags=%h",
               transactions, result, flags, expected, expected_flags);
      // Completion and output must survive held go and changing idle input data.
      left = 32'hdeadbeef; right = 32'h01234567;
      repeat (3) begin
        @(negedge clk);
        if (!done || result !== expected || flags !== expected_flags)
          $fatal(1, "result/flags not retained");
      end
      go = 0;
      repeat (gap) @(negedge clk);
      if (done) $fatal(1, "stale done after go release");
    end
  endtask

  initial begin
    repeat (3) @(negedge clk);
    reset = 0;
    repeat (3) begin
      check(0, 32'h3f800000, 32'h40000000, 32'h3f000000, 0, 1); // 1/2
      check(0, 32'h40400000, 32'h40000000, 32'h3fc00000, 0, 5); // 3/2
      check(0, 32'hbf800000, 32'h40000000, 32'hbf000000, 0, 1); // -1/2
      check(0, 32'h3f800000, 32'h41500000, 32'h3d9d89d9, 1, 2); // 1/13, inexact
      check(0, 32'h00000000, 32'h40000000, 32'h00000000, 0, 1); // zero
      check(0, 32'h80000000, 32'h40000000, 32'h80000000, 0, 1); // negative zero
      check(0, 32'h3f800000, 32'h00000000, 32'h7f800000, 8, 1); // divide by zero
      check(0, 32'h7f800000, 32'h40000000, 32'h7f800000, 0, 1); // infinity
      check(1, 32'h40800000, 32'h00000000, 32'h40000000, 0, 1); // sqrt(4)
      check(1, 32'h41100000, 32'h00000000, 32'h40400000, 0, 1); // sqrt(9)
    end
    if (accepted != transactions || completed != transactions)
      $fatal(1, "duplicate or missing request: accepted=%0d completed=%0d requests=%0d",
             accepted, completed, transactions);
    // Reset during an operation must cancel it and clear completion.
    @(negedge clk); left = 32'h3f800000; right = 32'h41500000; sqrtOp = 0; go = 1;
    repeat (3) @(negedge clk);
    reset = 1; go = 0;
    repeat (3) @(negedge clk);
    if (done || result !== 0 || flags !== 0) $fatal(1, "reset failed");
    reset = 0;
    check(0, 32'h40400000, 32'h40000000, 32'h3fc00000, 0, 1);
    main_go = 1;
    while (!main_done) @(negedge clk);
    if (main_a !== 32'h3f000000 || main_b !== 32'h3fc00000 || main_c !== 32'hbf000000)
      $fatal(1, "Calyx control returned %h %h %h", main_a, main_b, main_c);
    $display("PASS Calyx-generated consecutive division control");
    $display("PASS divider handshake: %0d checked results, including reset", transactions);
    $finish;
  end
endmodule
