<?xml version='1.0' encoding="utf-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:fo="http://www.w3.org/1999/XSL/Format"
                version='1.0'>

<!-- MJ 051122: patch of fop.xsl 1.10 to accommodate new FOP implementation of PDF bookmarks -->

<!-- ********************************************************************
     $Id: fop.xsl,v 1.10 2005/04/20 08:11:36 bobstayton Exp $
     ********************************************************************
     (c) Stephane Bline Peregrine Systems 2001
     Driver file to allow pdf bookmarking (based on fop implementation).
     ******************************************************************** -->
<!--
In PDF bookmarks can't be used characters with code>255. This version of file
translates characters with code>255 back to ASCII.

   Pavel Zampach (zampach@volny.cz)
-->

<xsl:variable name="a-dia" select=
"'&#257;&#259;&#261;&#263;&#265;&#267;&#269;&#271;&#273;&#275;&#277;&#279;&#281;&#283;&#339;&#285;&#287;&#289;&#291;&#293;&#295;&#297;&#299;&#301;&#303;&#305;&#309;&#311;&#314;&#316;&#318;&#320;&#322;&#324;&#326;&#328;&#331;&#333;&#335;&#337;&#341;&#343;&#345;&#347;&#349;&#351;&#353;&#355;&#357;&#359;&#361;&#363;&#365;&#367;&#369;&#371;&#373;&#375;&#378;&#380;&#382;&#256;&#258;&#260;&#262;&#264;&#266;&#268;&#270;&#272;&#274;&#276;&#278;&#280;&#282;&#338;&#284;&#286;&#288;&#290;&#292;&#294;&#296;&#298;&#300;&#302;&#304;&#308;&#310;&#313;&#315;&#317;&#319;&#321;&#323;&#325;&#327;&#330;&#332;&#334;&#336;&#340;&#342;&#344;&#346;&#348;&#350;&#352;&#354;&#356;&#358;&#360;&#362;&#364;&#366;&#368;&#370;&#372;&#374;&#376;&#377;&#379;&#381;'"/>
<xsl:variable name="a-asc" select=
"'aaaccccddeeeeeegggghhiiiiijklllllnnnnooorrrsssstttuuuuuuwyzzzAAACCCCDDEEEEEEGGGGHHIIIIIJKLLLLLNNNNOOORRRSSSSTTTUUUUUUWYYZZZ'"/>



<!-- Bookmarks for FOP version >= 0.90 -->
<!-- ================================= -->
<xsl:template match="*" mode="fop.outline"/>

<xsl:template match="set|book|part|reference|preface|chapter|appendix|article
                     |glossary|bibliography|index|setindex
                     |refentry
                     |sect1|sect2|sect3|sect4|sect5|section"
              mode="fop.outline">

  <xsl:variable name="id">
    <xsl:call-template name="object.id"/>
  </xsl:variable>
  <xsl:variable name="bookmark-label">
    <xsl:apply-templates select="." mode="object.title.markup"/>
  </xsl:variable>

 <xsl:choose>

    <xsl:when test="parent::*">
      <fo:bookmark internal-destination="{$id}" starting-state="hide">
        <fo:bookmark-title>
          <xsl:value-of select="normalize-space(translate($bookmark-label, $a-dia, $a-asc))"/>
        </fo:bookmark-title>
        <xsl:apply-templates select="*" mode="fop.outline"/>
      </fo:bookmark>
    </xsl:when>

    <xsl:otherwise>

      <fo:bookmark-tree>
      <fo:bookmark internal-destination="{$id}" starting-state="show">
        <fo:bookmark-title>
          <xsl:value-of select="normalize-space(translate($bookmark-label, $a-dia, $a-asc))"/>
        </fo:bookmark-title>
      </fo:bookmark>

      <xsl:variable name="toc.params">
        <xsl:call-template name="find.path.params">
          <xsl:with-param name="table" select="normalize-space($generate.toc)"/>
        </xsl:call-template>
      </xsl:variable>

      <xsl:if test="contains($toc.params, 'toc')
                    and (book|part|reference|preface|chapter|appendix|article
                         |glossary|bibliography|index|setindex
                         |refentry
                         |sect1|sect2|sect3|sect4|sect5|section)">
        <fo:bookmark internal-destination="toc...{$id}">
         <fo:bookmark-title>
            <xsl:call-template name="gentext">
              <xsl:with-param name="key" select="'TableofContents'"/>
            </xsl:call-template>
         </fo:bookmark-title>
        </fo:bookmark>
      </xsl:if>
      <xsl:apply-templates select="*" mode="fop.outline"/>

      </fo:bookmark-tree>

    </xsl:otherwise>

  </xsl:choose>

</xsl:template>


</xsl:stylesheet>

