<?xml version='1.0' encoding="utf-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                version="1.0">

    <!-- TOC -->
    <!-- * generate TOC only for the book not for Parts -->
    <xsl:param name="generate.toc" select="'book toc,title,figure,table'"/>

    <!-- BIBLIOGRAPHY -->
    <xsl:param name="bibliography.numbered" select="1"></xsl:param>
    <xsl:param name="bibliography.collection" select="'usersguide_bib.xml'"></xsl:param>

    <!-- ADMONITION -->
    <!-- * surround the body text with top and bottom solid lines
         * use Helvetica font for the admonition title
         * the title is centered -->
    <xsl:attribute-set name="admonition.properties">
      <xsl:attribute name="border-top-style">solid</xsl:attribute>
      <xsl:attribute name="border-bottom-style">solid</xsl:attribute>
    </xsl:attribute-set>

    <xsl:attribute-set name="admonition.title.properties">
      <xsl:attribute name="text-align">center</xsl:attribute>
      <xsl:attribute name="font-family">Helvetica</xsl:attribute>
    </xsl:attribute-set>

    <!-- FORMAL OBJECT TITLES (tables and figures) -->
    <!-- * customise title location -->
    <xsl:param name="formal.title.placement">
      figure after
      example before
      equation before
      table after
      procedure before
    </xsl:param>

    <!-- SECTIONS -->
    <xsl:param name="section.autolabel.max.depth" select="2"></xsl:param>

    <!-- Customise section labels -->
    <!-- font size is calculated dynamically by section.heading template -->
    <xsl:param name="section.label.includes.component.label" select="1"></xsl:param>

</xsl:stylesheet>
