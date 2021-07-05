CREATE TABLE `pfamNN` (
  `pfamA_acc` varchar(7) NOT NULL,
  PRIMARY KEY (`pfamA_acc`)
  ) ENGINE=MyISAM DEFAULT CHARSET=latin1;

SET GLOBAL LOCAL_INFILE=TRUE;
LOAD DATA LOCAL INFILE 'd:/Post-doc/Project_SoybeanPhospho/Orthologs/Pfam_SQL/pfamNN.txt' INTO TABLE pfamNN;