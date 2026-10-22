module std_divSqrtFN #(
    parameter expWidth = 8,
    parameter sigWidth = 24,
    parameter numWidth = 32
) (
    input clk,
    input reset,
    input go,
    input [(1 - 1):0] control,
    input sqrtOp,
    input [(expWidth + sigWidth - 1):0] left,
    input [(expWidth + sigWidth - 1):0] right,
    input [2:0] roundingMode,
    output logic [(expWidth + sigWidth - 1):0] out,
    output logic [4:0] exceptionFlags,
    output logic done
);

    logic busy;
    logic [(1 - 1):0] saved_control;
    wire [4:0] result_flags;
    wire request_valid = go && !busy && !done && !reset;

    // Intermediate signals for recoded formats
    wire [(expWidth + sigWidth):0] l_recoded, r_recoded;

    // Convert inputs from IEEE-754 to recoded format
    fNToRecFN #(expWidth, sigWidth) convert_l(
        .in(left),
        .out(l_recoded)
    );

    fNToRecFN #(expWidth, sigWidth) convert_r(
        .in(right),
        .out(r_recoded)
    );

    // Intermediate signals from `divSqrtRecFNToRaw_small`
    wire inReady, outValid;
    wire [2:0] roundingModeOut;
    wire invalidExc, infiniteExc, out_isNaN, out_isInf, out_isZero, out_sign;
    wire signed [(expWidth + 1):0] out_sExp;
    wire [(sigWidth + 2):0] out_sig;

    // Call HardFloat's `divSqrtRecFNToRaw_small` instead of `divSqrtRecFN_small` because the latter has signal duplicate declaration issue.
    divSqrtRecFNToRaw_small #(expWidth, sigWidth) divSqrtRecFNToRaw (
        .nReset(~reset),
        .clock(clk),
        .control(control),
        .inReady(inReady),
        .inValid(request_valid),
        .sqrtOp(sqrtOp),
        .a(l_recoded),
        .b(r_recoded),
        .roundingMode(roundingMode),
        .outValid(outValid),
        .sqrtOpOut(),  // Not connected
        .roundingModeOut(roundingModeOut),
        .invalidExc(invalidExc),
        .infiniteExc(infiniteExc),
        .out_isNaN(out_isNaN),
        .out_isInf(out_isInf),
        .out_isZero(out_isZero),
        .out_sign(out_sign),
        .out_sExp(out_sExp),
        .out_sig(out_sig)
    );

    // Intermediate signals for rounded result
    wire [(expWidth + sigWidth):0] res_recoded;

    // Round the output using HardFloat's rounding module
    roundRawFNToRecFN#(expWidth, sigWidth, 0) 
        roundRawOut (
            saved_control,
            invalidExc,
            infiniteExc,
            out_isNaN,
            out_isInf,
            out_isZero,
            out_sign,
            out_sExp,
            out_sig,
            roundingModeOut,
            res_recoded,
            result_flags
    );

    // Convert result back to IEEE754 format
    wire [(expWidth + sigWidth - 1):0] res_std;
    recFNToFN #(expWidth, sigWidth) convert_res(
        .in(res_recoded),
        .out(res_std)
    );

    // OASIS: one request per go/done transaction. No fixed-latency assumption.
    always_ff @(posedge clk) begin
        if (reset) begin
            busy <= 1'b0;
            done <= 1'b0;
            out <= '0;
            exceptionFlags <= '0;
            saved_control <= '0;
        end else begin
            if (!go) done <= 1'b0;
            if (request_valid && inReady) begin
                busy <= 1'b1;
                saved_control <= control;
            end
            if (busy && outValid) begin
                out <= res_std;
                exceptionFlags <= result_flags;
                done <= 1'b1;
                busy <= 1'b0;
            end
        end
    end
endmodule
