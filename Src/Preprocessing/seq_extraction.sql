SELECT 
n.pfamA_acc, CAST(q.sequence AS CHAR) AS sequence
FROM
    pfamnn n 
        JOIN
    pfama_reg_full_significant p ON p.pfamA_acc = n.pfamA_acc
        JOIN
    pfamseq q ON q.pfamseq_acc = p.pfamseq_acc